from collections import defaultdict
from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmengine.dataset import Compose
from mmseg.utils import register_all_modules

from mowing_terrain_seg.inference.backends import OnnxBackend, TorchBackend
from mowing_terrain_seg.inference.decode import (
    compute_confidence_unified,
    extract_mask_unified,
)
from mowing_terrain_seg.inference.thresholding import apply_confidence_threshold
from mowing_terrain_seg.inference import visualization as vis

register_all_modules()


class Backend(str, Enum):
    TORCH = "torch"
    ONNX = "onnx"


class BasePredictor:
    def __init__(
        self,
        cfg_uri: str,
        model_uri: str,
        backend: Backend,
        device: str = "cuda:0",
        conf_thresholds: Optional[Union[float, List[float]]] = None,
    ):
        self.cfg_uri = cfg_uri
        self.model_uri = model_uri
        self.backend = backend
        self.device = device
        self.conf_thresholds = conf_thresholds

        self._impl: Any = None
        self.cfg: Any = None
        self.model = None
        self.ort_session = None
        self.data_preprocessor = None
        self.num_classes: Optional[int] = None

        if backend == Backend.TORCH:
            impl = TorchBackend(cfg_uri, model_uri, device)
            self._impl = impl
            self.cfg = impl.cfg
            self.model = impl.model
            self.data_preprocessor = impl.data_preprocessor
            self.num_classes = impl.num_classes
        elif backend == Backend.ONNX:
            impl = OnnxBackend(cfg_uri, model_uri, device)
            self._impl = impl
            self.cfg = impl.cfg
            self.ort_session = impl.ort_session
            self.data_preprocessor = impl.data_preprocessor
            self.num_classes = impl.num_classes
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _prepare_data(
        self, imgs: Union[np.ndarray, Sequence[np.ndarray]]
    ) -> Tuple[Any, bool]:
        test_pipeline: list = []
        if self.backend == Backend.TORCH:
            test_pipeline = self.cfg.test_pipeline.copy()
            test_pipeline = [
                t for t in test_pipeline if t.get("type") != "LoadAnnotations"
            ]
        elif self.backend == Backend.ONNX:
            resize_cfg = self._impl.spec.resize_cfg
            scale = resize_cfg["size"]
            if isinstance(scale, (list, tuple)):
                scale = tuple(scale)
            test_pipeline = [
                {"type": "LoadImageFromNDArray"},
                {
                    "keep_ratio": resize_cfg["keep_ratio"],
                    "scale": scale,
                    "type": "Resize",
                },
                {"type": "PackSegInputs"},
            ]

        test_pipeline[0]["type"] = "LoadImageFromNDArray"

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
            data["inputs"].append(data_["inputs"])
            data["data_samples"].append(data_["data_samples"])

        return data, is_batch

    def _preprocess(self, data: Any) -> Any:
        return self.data_preprocessor(data, False)

    def _forward(self, data: Any) -> Any:
        return self._impl.forward(data)

    def _postprocess(self, raw_outputs: Any) -> Any:
        return raw_outputs

    def predict(
        self, imgs: Union[np.ndarray, Sequence[np.ndarray]]
    ) -> Any:
        data, is_batch = self._prepare_data(imgs)
        with torch.no_grad():
            preprocessed_data = self._preprocess(data)
            results = self._forward(preprocessed_data)
            raw_outputs = results if is_batch else results[0]
        return self._postprocess(raw_outputs)

    def __call__(self, imgs: Union[np.ndarray, Sequence[np.ndarray]]):
        return self.predict(imgs)


class SegPredictor(BasePredictor):
    def _postprocess(self, raw_outputs: Any) -> Any:
        is_batch = isinstance(raw_outputs, (list, tuple))
        n_cls = self.num_classes

        def one(raw: Any) -> np.ndarray:
            mask = extract_mask_unified(raw)
            if self.conf_thresholds is not None:
                conf = compute_confidence_unified(raw)
                mask = apply_confidence_threshold(
                    mask,
                    conf,
                    self.conf_thresholds,
                    n_cls,
                    raw,
                )
            return mask

        if is_batch:
            return [one(o) for o in raw_outputs]
        return one(raw_outputs)

    def get_auto_palette(self) -> List[List[int]]:
        return vis.get_auto_palette(self.num_classes or 0)

    def visualize_mask(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        opacity: float = 0.7,
        palette: Optional[
            Union[
                List[List[int]],
                List[Tuple[int, int, int]],
                List[str],
            ]
        ] = None,
    ) -> np.ndarray:
        return vis.visualize_mask(img, mask, opacity=opacity, palette=palette)
