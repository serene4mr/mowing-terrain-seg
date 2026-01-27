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
from mmseg.utils import register_all_modules

import onnxruntime as ort

from src.mowing_terrain_seg.utils.logger import LOGGER

register_all_modules()

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
        
        self.num_classes = None

        self._load_model()
        
    def _load_model(self) -> None:
        """
        Load model/engine from self.model_uri based on backend
        """
        if self.backend == Backend.TORCH:
            from mmengine.config import Config
            from mmseg.apis.inference import init_model
            
            self.cfg = Config.fromfile(self.cfg_uri)
            
            self.num_classes = self.cfg['num_classes']
            
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
            import json
            
            self.cfg = json.load(open(self.cfg_uri))
            self.num_classes = self.cfg['pipeline']['tasks'][2]['params']['num_classes']
            cfg_transforms = self.cfg['pipeline']['tasks'][0]['transforms']
            
            data_preprocessor_cfg = {}
            self.data_preprocessor = MODELS.build(
                dict(
                    {
                        'bgr_to_rgb': cfg_transforms[2]['to_rgb'],
                        'mean': cfg_transforms[2]['mean'],
                        'std': cfg_transforms[2]['std'],
                        'size': cfg_transforms[1]['size'],
                        'test_cfg': {'size_divisor': 32},
                        'seg_pad_val': 255,
                        'pad_val': 0,
                        'type': 'SegDataPreProcessor'
                    }
                )
            )
            self.data_preprocessor.to(self.device)
            
            if self.backend == Backend.ONNX:
                # Determine providers based on device setting
                if self.device and self.device.startswith('cuda'):
                    # Try CUDA first, ONNX Runtime will automatically fallback to CPU if needed
                    providers = ['CUDAExecutionProvider']
                else:
                    # Use CPU only
                    providers = ['CPUExecutionProvider']
                
                # Try to register custom ops if available (mmcv/mmdeploy)
                sess_options = None
                # Try mmcv first (common case)
                try:
                    from mmcv.ops import get_onnxruntime_op_path
                    import os
                    custom_op_path = get_onnxruntime_op_path()
                    if custom_op_path and os.path.exists(custom_op_path):
                        sess_options = ort.SessionOptions()
                        sess_options.register_custom_ops_library(custom_op_path)
                        LOGGER.info(f"Registered mmcv custom ops from: {custom_op_path}")
                except (ImportError, AttributeError, FileNotFoundError):
                    # Try mmdeploy alternative path
                    try:
                        import mmdeploy
                        import os.path as osp
                        # Common paths for mmdeploy custom ops
                        possible_paths = [
                            osp.join(osp.dirname(mmdeploy.__file__), 'lib', 'libmmdeploy_onnxruntime_ops.so'),
                            osp.join(osp.dirname(mmdeploy.__file__), 'lib', 'mmdeploy_onnxruntime_ops.so'),
                        ]
                        for custom_op_path in possible_paths:
                            if osp.exists(custom_op_path):
                                sess_options = ort.SessionOptions()
                                sess_options.register_custom_ops_library(custom_op_path)
                                LOGGER.info(f"Registered mmdeploy custom ops from: {custom_op_path}")
                                break
                    except (ImportError, FileNotFoundError):
                        pass  # Custom ops not available, will try standard loading
                
                try:
                    if sess_options is not None:
                        # Use custom ops if registered
                        self.ort_session = ort.InferenceSession(
                            self.model_uri, sess_options=sess_options, providers=providers)
                    else:
                        # Standard loading without custom ops
                        self.ort_session = ort.InferenceSession(
                            self.model_uri, providers=providers)
                except Exception as e:
                    error_msg = str(e)
                    # Check if error is due to custom operators (mmdeploy)
                    if 'grid_sampler' in error_msg or 'not a registered function/op' in error_msg:
                        raise RuntimeError(
                            f"Failed to load ONNX model: The model uses custom operators "
                            f"(e.g., mmdeploy's grid_sampler) that are not registered in standard ONNX Runtime. "
                            f"Original error: {error_msg}"
                        ) from e
                    else:
                        # Other errors (file not found, invalid model, etc.)
                        raise RuntimeError(
                            f"Failed to load ONNX model: {error_msg}"
                        ) from e
                
            elif self.backend == Backend.TENSORRT:
                #TODO: implement later
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
            test_pipeline = [
                {'type': 'LoadImageFromNDArray'},
                {
                    'keep_ratio': self.cfg['pipeline']['tasks'][0]['transforms'][1]['keep_ratio'], 
                    'scale': self.cfg['pipeline']['tasks'][0]['transforms'][1]['size'],
                    'type': 'Resize'
                },
                {'type': 'PackSegInputs'}
            ]

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
        elif self.backend == Backend.ONNX:
            # Get input name from ONNX session
            input_name = self.ort_session.get_inputs()[0].name
            # Convert tensor to numpy
            input_tensor = data['inputs']
            if isinstance(input_tensor, torch.Tensor):
                input_numpy = input_tensor.cpu().numpy()
            else:
                input_numpy = input_tensor
            # Create dict with input name as key
            ort_inputs = {input_name: input_numpy}
            ort_outputs = self.ort_session.run(None, ort_inputs)
            return ort_outputs
        elif self.backend == Backend.TENSORRT:
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
        """Extract a single mask from a MMSegmentation output or ONNX output.
        
        Args:
            raw_output: Single MMSegmentation data sample with pred_sem_seg attribute,
                or numpy array from ONNX inference
            
        Returns:
            np.ndarray: Segmentation mask as numpy array of shape (H, W)
        """
        # Handle ONNX outputs (numpy arrays)
        if isinstance(raw_output, np.ndarray):
            output = raw_output
            # Handle different output shapes
            if len(output.shape) == 4:
                # Shape: [batch, num_classes, H, W] or [batch, H, W]
                output = output[0]  # Remove batch dimension
            
            # After removing batch dimension, check remaining shape
            if len(output.shape) == 3:
                # Could be [1, H, W] (predictions with batch) or [num_classes, H, W] (logits)
                if output.shape[0] == 1:
                    # Shape: [1, H, W] - predictions with batch dimension, remove it
                    mask = output[0].astype(np.uint8)
                else:
                    # Shape: [num_classes, H, W] - logits, need argmax
                    mask = np.argmax(output, axis=0).astype(np.uint8)
            elif len(output.shape) == 2:
                # Shape: [H, W] - already predictions
                mask = output.astype(np.uint8)
            else:
                raise ValueError(f"Unexpected ONNX output shape: {output.shape}")
            return mask
        
        # Handle MMSegmentation data samples
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
            raw_output: MMSegmentation data sample with seg_logits attribute,
                or numpy array from ONNX inference
            
        Returns:
            np.ndarray or None: Confidence scores array of shape (H, W) with max probability 
                per pixel, or None if logits are not available
        """
        # Handle ONNX outputs (numpy arrays)
        if isinstance(raw_output, np.ndarray):
            output = raw_output
            # Handle different output shapes
            if len(output.shape) == 4:
                # Shape: [batch, num_classes, H, W]
                output = output[0]  # Remove batch dimension
            
            # After removing batch dimension, check remaining shape
            if len(output.shape) == 3:
                # Could be [1, H, W] (predictions) or [num_classes, H, W] (logits)
                if output.shape[0] == 1:
                    # Shape: [1, H, W] - predictions, can't compute confidence
                    return None
                else:
                    # Shape: [num_classes, H, W] - logits
                    # Convert to float32 for softmax computation
                    output_float = output.astype(np.float32)
                    # Convert to probabilities using softmax
                    logits_tensor = torch.from_numpy(output_float)
                    probs = F.softmax(logits_tensor, dim=0).numpy()
                    confidence_scores = np.max(probs, axis=0)
                    return confidence_scores
            else:
                # Shape: [H, W] - already predictions, can't compute confidence
                return None
        
        # Handle MMSegmentation data samples
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
            # Check if this is an ONNX output that's already predictions
            if isinstance(raw_output, np.ndarray):
                output_shape = raw_output.shape
                if len(output_shape) >= 2 and (len(output_shape) == 2 or 
                    (len(output_shape) == 3 and output_shape[0] == 1) or
                    (len(output_shape) == 4 and output_shape[1] == 1)):
                    LOGGER.warning(
                        f"Confidence thresholds provided but ONNX model outputs predictions "
                        f"(shape: {output_shape}), not logits. Thresholding requires logits. "
                        f"Please export ONNX model with logits output (before ArgMax) to enable thresholding."
                    )
                else:
                    LOGGER.warning(
                        f"Confidence thresholds provided but could not extract logits from output "
                        f"(shape: {output_shape}). Thresholding will be skipped."
                    )
            else:
                LOGGER.warning(
                    "Confidence thresholds provided but no seg_logits found in model output. "
                    "Thresholding will be skipped."
                )
            return mask
        
        # Ensure mask dtype can handle 255 (uint8 or larger)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        # Normalize thresholds: if single value, apply to all classes
        if isinstance(self.conf_thresholds, (int, float)):
            thresholds = [self.conf_thresholds] * self.num_classes
        else:
            thresholds = self.conf_thresholds
        
        # Create filtered mask
        filtered_mask = mask.copy()
        
        # Apply threshold for each class
        for class_id, threshold in enumerate(thresholds):
            if class_id >= self.num_classes:
                continue
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
    
    def get_auto_palette(self) -> List[List[int]]:
        """Generates a high-contrast palette based on self.num_classes using HSV color wheel.
        
        Returns:
            List[List[int]]: A list of RGB color lists [[R, G, B], ...]
        """
        import cv2
        
        if self.num_classes is None or self.num_classes <= 0:
            return []
            
        # Generate colors in HSV space for maximum distinctness
        # Start at Hue 30 (Yellow/Orange) to avoid Red (0), and use high Value to avoid Black
        hsv_colors = np.zeros((self.num_classes, 1, 3), dtype=np.uint8)
        hsv_colors[:, 0, 0] = np.linspace(30, 179, self.num_classes, endpoint=False)
        hsv_colors[:, 0, 1] = 200 
        hsv_colors[:, 0, 2] = 255
        
        # Convert to BGR (OpenCV standard)
        bgr_palette = cv2.cvtColor(hsv_colors, cv2.COLOR_HSV2BGR).reshape(-1, 3)
        
        # Convert BGR to RGB for consistency with visualize_mask expectations
        rgb_palette = bgr_palette[:, [2, 1, 0]].tolist()
        
        return rgb_palette
    
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
        
        # Ensure spatial dimensions match (resize mask if necessary)
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
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
        
    