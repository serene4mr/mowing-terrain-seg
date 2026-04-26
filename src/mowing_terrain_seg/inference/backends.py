"""Torch / ONNX inference backends for segmentation."""

from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.registry import MODELS

from mmseg.apis.inference import init_model
from mmseg.utils import register_all_modules

from mowing_terrain_seg.inference.deploy_pipeline import DeployPipelineSpec

register_all_modules()


def _init_model_legacy_checkpoint(
    config: Any, checkpoint: str, device: str
) -> Any:
    """Load weights with ``weights_only=False`` for PyTorch 2.4+ / checkpoints with pickle."""
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    try:
        torch.load = patched_load
        return init_model(
            config=config, checkpoint=checkpoint, device=device
        )
    finally:
        torch.load = original_load


class TorchBackend:
    def __init__(self, cfg_uri: str, model_uri: str, device: str):
        self.device = device
        self.cfg = Config.fromfile(cfg_uri)
        self.model = _init_model_legacy_checkpoint(
            self.cfg, model_uri, device
        )
        self.num_classes = int(self.cfg["num_classes"])
        dp = self.cfg["data_preprocessor"]
        self.data_preprocessor = MODELS.build(dp)
        self.data_preprocessor.to(device)

    def build_compose(self, test_pipeline) -> Compose:
        return Compose(test_pipeline)

    def forward(self, data: Any) -> Any:
        return self.model._run_forward(data, mode="predict")


class OnnxBackend:
    def __init__(self, cfg_uri: str, model_uri: str, device: str):
        self.device = device
        with open(cfg_uri, encoding="utf-8") as f:
            import json

            self.cfg_dict = json.load(f)
        self.spec: DeployPipelineSpec = DeployPipelineSpec.from_dict(
            self.cfg_dict
        )
        self.num_classes = self.spec.num_classes
        t = self.spec.cfg_transforms
        self.data_preprocessor = MODELS.build(
            dict(
                bgr_to_rgb=t[2]["to_rgb"],
                mean=t[2]["mean"],
                std=t[2]["std"],
                size=t[1]["size"],
                test_cfg={"size_divisor": 32},
                seg_pad_val=255,
                pad_val=0,
                type="SegDataPreProcessor",
            )
        )
        self.data_preprocessor.to(device)

        import onnxruntime as ort  # lazy: optional dep for ONNX backend

        if device and device.startswith("cuda"):
            providers: List[str] = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        try:
            self.ort_session = ort.InferenceSession(
                model_uri, providers=providers
            )
        except Exception as e:
            error_msg = str(e)
            if (
                "grid_sampler" in error_msg
                or "not a registered function/op" in error_msg
            ):
                raise RuntimeError(
                    "Failed to load ONNX model: The model uses custom operators "
                    "(e.g., mmdeploy's grid_sampler) that are not registered in standard ONNX Runtime. "
                    f"Original error: {error_msg}"
                ) from e
            raise RuntimeError(f"Failed to load ONNX model: {error_msg}") from e

    @property
    def cfg(self) -> Union[Config, dict]:
        return self.cfg_dict

    def forward(self, data: Any) -> Any:
        input_name = self.ort_session.get_inputs()[0].name
        input_tensor = data["inputs"]
        if isinstance(input_tensor, torch.Tensor):
            input_numpy = input_tensor.cpu().numpy()
        else:
            input_numpy = input_tensor
        return self.ort_session.run(
            None, {input_name: input_numpy}
        )
