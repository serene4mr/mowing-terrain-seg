from collections import defaultdict
from typing import Any, Union, Sequence
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
        
        print(self.backend)
        
        self._load_model()
        
    def __call__(self):
        pass
        
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
            torch.load = patched_load
            
            self.model = init_model(
                config = self.cfg,
                checkpoint=self.model_uri,
                device=self.device
            )
            
        if self.backend == Backend.ONNX or self.backend == Backend.TENSORRT:
            # TODO: implement later
            raise NotImplementedError
        
    
    def _prepare_data(
        self, 
        imgs: Union[np.ndarray, Sequence[np.ndarray]]
    ) -> tuple(Union[dict, tuple, list], bool) :
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
            
            test_pipeline = self.cfg.test_pipeline
            for t in test_pipeline:
                if t.get('type') == 'LoadAnnotations':
                    test_pipeline.remove(t)
        
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
        
        if self.backend == Backend.TORCH:
            data_preprocessor_cfg = self.cfg['data_preprocessor']
            
        if self.backend == Backend.ONNX or self.backend == Backend.TENSORRT:
            # TODO: implement later with json input
            raise NotImplementedError
        
        data_preprocessor = MODELS.build(data_preprocessor_cfg)
        
        return data_preprocessor(data, False)
        

    # @abstractmethod
    # def _forward(self, batch: Any) -> Any:
    #     """forward data through backend (Torch / ONNX / TensorRT)."""
    #     ...
        
    # @abstractmethod
    # def _postprocess(self, raw_output: Any):
    #     """
    #     Convert raw output of prediction to final results (mask, v.v).
    #     """
    #     ...   
             
    
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
        batch = self._preprocess(imgs)
        raw = self._forward(batch)
        outputs = self._postprocess(raw)
        return outputs
    

class SegPredictor(BasePredictor):
    
    def __init__(self, cfg_uri: str, model_uri: str, backend: Backend, device: str ='cuda:0'):
        super().__init__(cfg_uri, model_uri, backend, device)
        
        
        
        
    
        
        
        
    