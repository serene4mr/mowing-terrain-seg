"""Parse MMDeploy ``pipeline.json`` for inference (no magic task indices)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DeployPipelineSpec:
    """Preprocess + task params extracted from mmdeploy pipeline config."""

    num_classes: int
    resize_cfg: Dict[str, Any]
    cfg_transforms: List[Dict[str, Any]]

    @classmethod
    def from_uri(cls, cfg_uri: str) -> "DeployPipelineSpec":
        with open(cfg_uri, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, pipeline_json: Dict[str, Any]) -> "DeployPipelineSpec":
        tasks = pipeline_json["pipeline"]["tasks"]
        t0_transforms = tasks[0]["transforms"]

        resize_cfg: Optional[Dict[str, Any]] = None
        for t in t0_transforms:
            if t.get("type") == "Resize":
                resize_cfg = t
                break
        if resize_cfg is None:
            if len(t0_transforms) > 1:
                resize_cfg = t0_transforms[1]
            else:
                raise ValueError(
                    "No Resize transform found in pipeline tasks[0]['transforms']"
                )

        num_classes: Optional[int] = None
        for task in tasks:
            params = task.get("params") or {}
            if "num_classes" in params:
                num_classes = params["num_classes"]
                break
        if num_classes is None:
            num_classes = tasks[2]["params"]["num_classes"]

        return cls(
            num_classes=num_classes,
            resize_cfg=resize_cfg,
            cfg_transforms=t0_transforms,
        )
