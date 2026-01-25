from collections import defaultdict
from typing import Any, Union, Sequence, Tuple, Optional, List
from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageColor

import mmseg
from mmengine.dataset import Compose
from mmengine.registry import MODELS

class Backend(str, Enum):
    TORCH = "torch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"

class BasePredictor:
    def __init__(
        self, 
        cfg_uri: str, 
        model_uri: str, 
        backend: Backend, 
        device: str = 'cuda:0',
        conf_thresholds: Optional[Union[float, List[float]]] = None
    ):
        """
        Initialize BasePredictor.
        
        Args:
            cfg_uri: Path to model configuration file
            model_uri: Path to model checkpoint file
            backend: Backend type (TORCH, ONNX, TENSORRT)
            device: Device to run inference on (default: 'cuda:0')
            conf_thresholds: Optional confidence threshold(s) per class.
                Can be:
                - A single float: applies the same threshold to all classes
                - A list of floats: applies per-class thresholds
                If provided, predictions with confidence below threshold for their class
                will be filtered. Implementation is task-specific and handled by subclasses.
        """
        
        self.cfg_uri = cfg_uri
        self.model_uri = model_uri # checkpoint path in term of torch backend
        self.backend = backend
        self.device = device
        self.conf_thresholds = conf_thresholds
        
        self.model = None # placeholder for model
        self.cfg = None # placeholder for cfg

        self._load_model()
        
    def _load_model(self) -> None:
        """
        Load model/engine from self.model_uri based on backend
        """
        if self.backend == Backend.TORCH:
            from mmengine.config import Config
            from mmseg.apis.inference import init_model
            
            self.cfg = Config.fromfile(self.cfg_uri)
            
            # Temporarily monkey patch torch.load to use weights_only=False
            original_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)

            try:
                torch.load = patched_load
            
                self.model = init_model(
                    config = self.cfg,
                    checkpoint=self.model_uri,
                    device=self.device
                )
            finally:
                torch.load = original_load
                
            data_preprocessor_cfg = self.cfg['data_preprocessor']
            self.data_preprocessor = MODELS.build(data_preprocessor_cfg)
            self.data_preprocessor.to(self.device)
            
        if self.backend == Backend.ONNX or self.backend == Backend.TENSORRT:
            # TODO: implement later
            raise NotImplementedError
        
    
    def _prepare_data(
        self, 
        imgs: Union[np.ndarray, Sequence[np.ndarray]]
    ) -> Tuple[Union[dict, tuple, list], bool]:
        """
        Prepare input images by applying test pipeline transforms.
        
        This method sets up the appropriate test pipeline based on the backend,
        applies image transforms (e.g., Resize, LoadImageFromNDArray), and 
        structures the data for further processing. Note that model-specific 
        preprocessing (normalization, tensor conversion) is handled separately 
        in _preprocess().
        
        Args:
            imgs: Input image(s) to prepare. Can be:
                - np.ndarray: Single image as a numpy array (HWC format)
                - Sequence[np.ndarray]: List or tuple of numpy arrays for batch processing
        
        Returns:
            tuple: A tuple containing:
                - data (dict): Processed data dictionary with 'inputs' and 'data_samples' keys
                - is_batch (bool): True if input was a batch (list/tuple), False if single image
        """
        
        test_pipeline = []
        if self.backend == Backend.TORCH:
            test_pipeline = self.cfg.test_pipeline.copy()  # Copy to avoid modifying original
            # Filter out LoadAnnotations transforms
            test_pipeline = [t for t in test_pipeline if t.get('type') != 'LoadAnnotations']
        
        if self.backend == Backend.ONNX or self.backend == Backend.TENSORRT:
            # test_pipeline = [
            #     {'type': 'LoadImageFromNDArray'},
            #     {'keep_ratio': True, 'scale': (1024, 544), 'type': 'Resize'},
            #     {'type': 'PackSegInputs'}
            # ]
            # TODO: implement later
            raise NotImplementedError

        test_pipeline[0]['type'] = 'LoadImageFromNDArray'

        is_batch = True
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
            is_batch = False

        pipeline = Compose(test_pipeline)

        data = defaultdict(list)
        for img in imgs:
            if isinstance(img, np.ndarray):
                data_ = dict(img=img)
            else:
                data_ = dict(img_path=img)
            data_ = pipeline(data_)
            data['inputs'].append(data_['inputs'])
            data['data_samples'].append(data_['data_samples'])

        return data, is_batch
        
    def _preprocess(self, data: Union[dict, tuple, list]) -> Union[dict, tuple, list]:
        """
        Apply model-specific preprocessing to prepared data.
        
        This method applies the data preprocessor (normalization, tensor conversion,
        channel reordering, etc.) to data that has been prepared by _prepare_data().
        The data preprocessor is built from the model configuration and handles
        the final transformations needed before model inference.
        
        Args:
            data: Prepared data from _prepare_data(). Can be:
                - dict: Data dictionary with 'inputs' and 'data_samples' keys
                - tuple: Tuple of (inputs, data_samples)
                - list: List of data items
        
        Returns:
            Preprocessed data ready for model inference. Format matches input format
            (dict, tuple, or list) but with normalized tensors and proper formatting.
        """
        
        preprocessed_data = self.data_preprocessor(data, False)
        
        return preprocessed_data
        

    def _forward(self, data: Union[dict, tuple, list]) -> list:
        """forward data through backend (Torch / ONNX / TensorRT)."""
        
        if self.backend == Backend.TORCH:
            out_data = self.model._run_forward(data, mode='predict')
            return out_data
        elif self.backend == Backend.ONNX or self.backend == Backend.TENSORRT:
            # TODO: implement later
            raise NotImplementedError(f"Backend {self.backend} not yet implemented")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        

    def _postprocess(self, raw_outputs: Any):
        """
        Convert raw output of prediction to final results (mask, v.v).
        """
        return raw_outputs
             
    def predict(self, imgs: Union[np.ndarray, Sequence[np.ndarray]]):
        """
        Run inference on one or a batch of images

        Args:
            imgs:
            - np.ndarray: single image
            - Sequence[np.ndarray]: list/tuple of images
            
        Returns:
            - output for single input, or list of outputs for batch
        """
        
        data, is_batch = self._prepare_data(imgs)
        
        with torch.no_grad():
            preprocessed_data = self._preprocess(data)
            results = self._forward(preprocessed_data)
            raw_outputs = results if is_batch else results[0]
            
        outputs = self._postprocess(raw_outputs)
        return outputs
    
    def __call__(self, imgs: Union[np.ndarray, Sequence[np.ndarray]]):
        return self.predict(imgs)
    
    
class SegPredictor(BasePredictor):
    
    def __init__(
        self, 
        cfg_uri: str, 
        model_uri: str, 
        backend: Backend, 
        device: str = 'cuda:0',
        conf_thresholds: Optional[Union[float, List[float]]] = None
    ):
        """
        Initialize SegPredictor with optional confidence thresholds.
        
        Args:
            cfg_uri: Path to model configuration file
            model_uri: Path to model checkpoint file
            backend: Backend type (TORCH, ONNX, TENSORRT)
            device: Device to run inference on (default: 'cuda:0')
            conf_thresholds: Optional confidence threshold(s) per class.
                Can be:
                - A single float: applies the same threshold to all classes
                - A list of floats: applies per-class thresholds
                If provided, pixels with confidence below threshold for their class
                will be set to 255 (indicating uncertain/filtered pixels).
        """
        super().__init__(cfg_uri, model_uri, backend, device, conf_thresholds)
        
    def _postprocess(self, raw_outputs):
        """Extract segmentation masks from raw MMSegmentation outputs.
        
        This method handles both single and batch outputs, extracts masks,
        and optionally applies confidence thresholding.
        
        Args:
            raw_outputs: MMSegmentation data sample(s) containing prediction results.
                Can be a single data sample or a list of data samples.
            
        Returns:
            np.ndarray or list: Segmentation mask(s) as numpy array(s) of shape (H, W)
                with class indices. Returns single array for single input, list for batch.
        """
        # Check if batch (list) or single output
        is_batch = isinstance(raw_outputs, (list, tuple))
        
        if is_batch:
            # Process each output in the batch
            masks = []
            for output in raw_outputs:
                mask = self._extract_single_mask(output)
                if self.conf_thresholds is not None:
                    mask = self._apply_confidence_threshold(mask, output)
                masks.append(mask)
            return masks
        else:
            # Process single output
            mask = self._extract_single_mask(raw_outputs)
            if self.conf_thresholds is not None:
                mask = self._apply_confidence_threshold(mask, raw_outputs)
            return mask
    
    def _extract_single_mask(self, raw_output):
        """Extract a single mask from a MMSegmentation output.
        
        Args:
            raw_output: Single MMSegmentation data sample with pred_sem_seg attribute
            
        Returns:
            np.ndarray: Segmentation mask as numpy array of shape (H, W)
        """
        if hasattr(raw_output, 'pred_sem_seg') and raw_output.pred_sem_seg is not None:
            mask = raw_output.pred_sem_seg.data.cpu().numpy()
            # Remove batch dimension if present (shape: [1, H, W] -> [H, W])
            if len(mask.shape) == 3:
                mask = mask[0]
            return mask
        elif hasattr(raw_output, 'pred_instances'):
            # Handle instance segmentation outputs
            # For now, return the raw output - can be extended later
            return raw_output
        else:
            raise ValueError(
                f"No segmentation mask found in output. "
                f"Available attributes: {dir(raw_output)}"
            )
    
    def _compute_confidence_scores(self, raw_output):
        """Compute confidence scores from logits.
        
        Args:
            raw_output: MMSegmentation data sample with seg_logits attribute
            
        Returns:
            np.ndarray or None: Confidence scores array of shape (H, W) with max probability 
                per pixel, or None if logits are not available
        """
        if not hasattr(raw_output, 'seg_logits') or raw_output.seg_logits is None:
            return None
        
        # Extract logits and convert to probabilities
        logits = raw_output.seg_logits.data.cpu().numpy()
        # logits shape is [num_classes, height, width] - no batch dimension
        
        # Convert to probabilities using softmax
        logits_tensor = torch.from_numpy(logits)
        probs = F.softmax(logits_tensor, dim=0).numpy()
        confidence_scores = np.max(probs, axis=0)
        
        return confidence_scores
    
    def _apply_confidence_threshold(self, mask: np.ndarray, raw_output) -> np.ndarray:
        """Apply confidence threshold filtering to segmentation mask.
        
        Args:
            mask: Segmentation mask as numpy array (H, W) with class indices
            raw_output: MMSegmentation data sample with seg_logits attribute
            
        Returns:
            np.ndarray: Filtered mask with low-confidence pixels set to 255.
                255 indicates uncertain/filtered pixels that should be rendered as black.
        """
        if self.conf_thresholds is None:
            return mask
        
        # Get confidence scores
        confidence_scores = self._compute_confidence_scores(raw_output)
        if confidence_scores is None:
            # No logits available, return mask unchanged
            return mask
        
        # Ensure mask dtype can handle 255 (uint8 or larger)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        # Normalize thresholds: if single value, apply to all classes
        if isinstance(self.conf_thresholds, (int, float)):
            max_class_id = int(mask.max()) if mask.size > 0 else 0
            thresholds = [self.conf_thresholds] * (max_class_id + 1)
        else:
            thresholds = self.conf_thresholds
        
        # Create filtered mask
        filtered_mask = mask.copy()
        
        # Apply threshold for each class
        for class_id, threshold in enumerate(thresholds):
            if class_id > mask.max():
                continue  # Skip if class_id doesn't exist in mask
            class_mask = (mask == class_id)
            low_confidence = (confidence_scores < threshold) & class_mask
            filtered_mask[low_confidence] = 255  # Mark as uncertain/filtered
        
        return filtered_mask
    
    def _extract_masks(self, raw_outputs):
        """Extract numpy mask arrays from MMSegmentation outputs.
        
        This is a legacy method for backward compatibility. Use _postprocess instead.
        
        Args:
            raw_outputs: MMSegmentation data sample(s) with pred_sem_seg attribute
            
        Returns:
            np.ndarray or list: Segmentation mask(s) as numpy array(s)
        """
        # Handle batch case
        if isinstance(raw_outputs, (list, tuple)):
            return [self._extract_single_mask(output) for output in raw_outputs]
        else:
            return self._extract_single_mask(raw_outputs)
    
    def get_mask_array(self, raw_outputs):
        """Public API: Get mask as numpy array from raw outputs.
        
        Args:
            raw_outputs: MMSegmentation data sample(s)
            
        Returns:
            np.ndarray: Segmentation mask as numpy array
        """
        return self._extract_masks(raw_outputs)
    
    def visualize_mask(
        self, 
        img: np.ndarray, 
        mask: np.ndarray, 
        opacity: float = 0.7,
        palette: Optional[Union[List[List[int]], List[Tuple[int, int, int]], List[str]]] = None
    ) -> np.ndarray:
        """Create overlay visualization of mask on image with class-specific colors.
        
        Args:
            img: Original image as numpy array (H, W, 3) in BGR format
            mask: Segmentation mask as numpy array (H, W) with class indices
            opacity: Overlay opacity between 0.0 and 1.0
            palette: Optional list of colors for each class ID. Can be:
                - List of RGB tuples/lists: [[R, G, B], [R, G, B], ...]
                - List of color name strings: ['white', 'gray', 'green', ...]
                Color name strings must be supported by PIL.ImageColor.
                If None, uses default green color for all non-background pixels.
            
        Returns:
            np.ndarray: Overlay image with mask visualization
        """
        import cv2
        
        # Ensure mask is uint8
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        # Create colored mask from palette
        if palette is not None:
            # Convert palette to BGR colormap array
            max_class_id = int(mask.max())
            num_classes = len(palette)
            
            # Create colormap array (handle case where mask has more classes than palette)
            colormap_array = np.zeros((max(max_class_id + 1, num_classes), 3), dtype=np.uint8)
            
            for class_id, color in enumerate(palette):
                if class_id < len(colormap_array):
                    # Check if color is a string (color name) or list/tuple (RGB)
                    if isinstance(color, str):
                        # Convert color name string to RGB using PIL ImageColor
                        rgb = ImageColor.getrgb(color)
                    else:
                        # Assume it's a list or tuple of RGB values
                        rgb = tuple(color)
                    
                    # Raise warning if palette contains black (reserved for background)
                    if rgb == (0, 0, 0):
                        import logging
                        logging.warning(
                            f"Class ID {class_id} is assigned black (0, 0, 0) in the palette. "
                            f"This conflicts with the reserved background/filtered color."
                        )

                    # Convert RGB to BGR for OpenCV
                    colormap_array[class_id] = [rgb[2], rgb[1], rgb[0]]
            
            # Apply colormap: colored_mask[class_id] = colormap_array[class_id]
            colored_mask = colormap_array[mask]
        else:
            # Default: green for all valid classes (excluding filtered pixels 255)
            colored_mask = np.zeros_like(img)
            # Only color pixels that are not filtered (255). 
            # Note: class 0 is treated as a valid class.
            valid_pixels = (mask < 255)
            colored_mask[valid_pixels] = [0, 255, 0]  # Green overlay for valid classes
        
        # Blend with original image
        overlay = cv2.addWeighted(img, 1 - opacity, colored_mask, opacity, 0)
        return overlay
        
    