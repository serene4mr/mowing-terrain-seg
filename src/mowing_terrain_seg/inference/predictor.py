from collections import defaultdict
from typing import Any, Union, Sequence, Tuple
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import torch

import mmseg
from mmengine.dataset import Compose
from mmengine.registry import MODELS

class Backend(str, Enum):
    TORCH = "torch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"

class BasePredictor(ABC):
    def __init__(self, cfg_uri: str, model_uri: str, backend: Backend, device: str = 'cuda:0'):
        """
        
        """
        
        self.cfg_uri = cfg_uri
        self.model_uri = model_uri # checkpoint path in term of torch backend
        self.backend = backend
        self.device = device
        
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
    
    def __init__(self, cfg_uri: str, model_uri: str, backend: Backend, device: str ='cuda:0'):
        super().__init__(cfg_uri, model_uri, backend, device)
        
    def _postprocess(self, raw_outputs):
        """Extract segmentation masks from raw MMSegmentation outputs.
        
        Args:
            raw_outputs: MMSegmentation data sample(s) containing prediction results
            
        Returns:
            np.ndarray or list: Segmentation mask(s) as numpy array(s) of shape (H, W)
                with class indices. Returns single array for single input, list for batch.
        """
        return self._extract_masks(raw_outputs)
    
    def _extract_masks(self, raw_outputs):
        """Extract numpy mask arrays from MMSegmentation outputs.
        
        Args:
            raw_outputs: MMSegmentation data sample(s) with pred_sem_seg attribute
            
        Returns:
            np.ndarray or list: Segmentation mask(s) as numpy array(s)
        """
        # Handle single output (data sample)
        if hasattr(raw_outputs, 'pred_sem_seg') and raw_outputs.pred_sem_seg is not None:
            mask = raw_outputs.pred_sem_seg.data.cpu().numpy()
            # Remove batch dimension if present (shape: [1, H, W] -> [H, W])
            if len(mask.shape) == 3:
                mask = mask[0]
            return mask
        elif hasattr(raw_outputs, 'pred_instances'):
            # Handle instance segmentation outputs
            # For now, return the raw output - can be extended later
            return raw_outputs
        else:
            raise ValueError(
                f"No segmentation mask found in output. "
                f"Available attributes: {dir(raw_outputs)}"
            )
    
    def get_mask_array(self, raw_outputs):
        """Public API: Get mask as numpy array from raw outputs.
        
        Args:
            raw_outputs: MMSegmentation data sample(s)
            
        Returns:
            np.ndarray: Segmentation mask as numpy array
        """
        return self._extract_masks(raw_outputs)
    
    def visualize_mask(self, img: np.ndarray, mask: np.ndarray, opacity: float = 0.7) -> np.ndarray:
        """Create overlay visualization of mask on image.
        
        Args:
            img: Original image as numpy array (H, W, 3) in BGR format
            mask: Segmentation mask as numpy array (H, W) with class indices
            opacity: Overlay opacity between 0.0 and 1.0
            
        Returns:
            np.ndarray: Overlay image with mask visualization
        """
        import cv2
        
        # Create colored mask (you may want to use a colormap here)
        # For now, simple visualization
        colored_mask = np.zeros_like(img)
        colored_mask[mask > 0] = [0, 255, 0]  # Green overlay for non-background
        
        # Blend with original image
        overlay = cv2.addWeighted(img, 1 - opacity, colored_mask, opacity, 0)
        return overlay
        
    