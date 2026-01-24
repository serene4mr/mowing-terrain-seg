from typing import Any, Literal, Union, Sequence
from abc import ABC, abstractmethod

import numpy as np
import torch

class Backend(str, Enum):
    TORCH = "torch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"

class BasePredictor(ABC):
    def __init__(self, cfg_uri: dict, model_uri: str, backend: Backend):
        """
        
        """
        
        self.cfg_uri = cfg_uri
        self.model_uri = model_uri
        self.backend =  backend
        
    def load_model(self) -> None:
        """
        Load model/engine from self.model_uri based on backend
        """
        from mmengine.config import Config
        

    @abstractmethod
    def _preprocess(
        self,
        images: Union[np.ndarray, Sequence[np.ndarray]],
    ) -> Any:
        """Convert raw images (HWC np.ndarray) -> batch input for backend."""
        ...

    @abstractmethod
    def _forward(self, batch: Any) -> Any:
        """forward data through backend (Torch / ONNX / TensorRT)."""
        ...
        
    @abstractmethod
    def _postprocess(self, raw_output: Any):
        """
        Convert raw output of prediction to final results (mask, v.v).
        """
        ...   
             
    
    def predict(self, images: Union[np.ndarray, Sequence[np.ndarray]]):
        """
        Run inference on one or a batch of images

        Args:
            images:
            - np.ndarray: single image
            - Sequence[np.ndarray]: list/tuple of images
            
        Returns:
            - output for single input, or list of outputs for batch
        """
        batch = self._preprocess(images)
        raw = self._forward(batch)
        outputs = self._postprocess(raw)
        return outputs
    

class SegPredictor(BasePredictor):
    
    def __init__(self, cfg: dict, model_uri: str, backend: Backend):
        super().__init__(cfg, model_uri, backend)
        
        
        
        
    
        
        
        
    