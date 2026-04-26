# Copyright (c) OpenMMLab. Mowing-terrain-seg. SPDX-License-Identifier: Apache-2.0
"""Hugging Face model card (README) text from a release summary."""

from __future__ import annotations

from typing import Any, Dict


def _classes_str(
    summary: Dict[str, Any]
) -> str:
    """Human-readable class list for the card."""
    ds = summary.get("dataset")
    if not isinstance(
        ds,
        dict
    ):
        return "(see `config.py` in this repo for class names / metainfo.)"
    cl = ds.get("classes")
    if cl is None:
        return "(see `config.py` in this repo for class names / metainfo.)"
    if isinstance(
        cl,
        (list, tuple)
    ):
        return ", ".join(
            str(c) for c in cl
        )
    return str(cl)


def render(
    *,
    summary: Dict[str, Any],
    repo_id: str,
    tag: str,
    git_sha: str,
    message: str,
    has_onnx: bool,
) -> str:
    name = (
        repo_id.split("/", 1)[-1]
        if "/" in repo_id
        else repo_id
    )
    miou_line = "            value: null"
    best = summary.get("best")
    if isinstance(
        best,
        dict
    ):
        m = best.get("metrics")
        if isinstance(
            m,
            dict
        ) and "mIoU" in m:
            try:
                v = float(
                    m["mIoU"]
                )
                miou_line = f"            value: {v}"
            except (TypeError, ValueError):
                miou_line = f"            value: {m['mIoU']}"
    ex = summary.get("experiment", {})
    if not isinstance(
        ex,
        dict
    ):
        ex = {}
    dstype = "ycor"
    ds = summary.get("dataset")
    if isinstance(
        ds,
        dict
    ) and "type" in ds:
        dstype = str(ds.get("type", "ycor"))
    front = f"""---
license: apache-2.0
library_name: mmsegmentation
tags:
  - semantic-segmentation
  - off-road
  - autonomous-mowing
datasets:
  - {dstype}
metrics:
  - mIoU
model-index:
  - name: {name}
    results:
      - task:
          type: image-segmentation
        dataset:
          name: YCOR-val
          type: ycor
        metrics:
          - type: mIoU
{miou_line}
---

# {name} ({tag})

{message or "(no message)"}

## Provenance
- Code repo: commit ``{git_sha}``
- Config SHA-256: ``{ex.get("config_sha256", "n/a")}`` (local run config file)
- Trained (max step in scalars): {ex.get("iterations", "n/a")}
- Run started: {ex.get("started_at", "n/a")}

## Classes
{_classes_str(summary)}

## Usage (PyTorch)

This tag was produced with **mowing-terrain-seg**. Install that repo, register
custom modules, then load ``config.py`` and ``pytorch/best.pth`` from this Hub
revision.

```python
from huggingface_hub import hf_hub_download
import mowing_terrain_seg
from mowing_terrain_seg.inference.predictor import SegPredictor, Backend

mowing_terrain_seg.register_all()

REPO = "{repo_id}"
REV = "{tag}"

cfg  = hf_hub_download(REPO, "config.py",        revision=REV)
ckpt = hf_hub_download(REPO, "pytorch/best.pth", revision=REV)
pred = SegPredictor(cfg_uri=cfg, model_uri=ckpt, backend=Backend.TORCH)
mask = pred.predict("input.jpg")
```

"""
    return front + _onnx_block(
        repo_id, tag, has_onnx
    ) + _files_block()


def _onnx_block(
    repo_id: str, tag: str, has_onnx: bool
) -> str:
    if not has_onnx:
        return "\n**ONNX:** This tag does not include an ``onnx/`` directory.\n\n"
    return f"""
## Usage (ONNX / ONNX Runtime)

```python
from huggingface_hub import snapshot_download
import mowing_terrain_seg
from mowing_terrain_seg.inference.predictor import SegPredictor, Backend

mowing_terrain_seg.register_all()

REPO = "{repo_id}"
REV  = "{tag}"
local = snapshot_download(REPO, revision=REV, allow_patterns=["onnx/*"])
pred = SegPredictor(
    cfg_uri=f"{{local}}/onnx/pipeline.json",
    model_uri=f"{{local}}/onnx/end2end.onnx",
    backend=Backend.ONNX,
)
mask = pred.predict("input.jpg")
```

"""


def _files_block() -> str:
    return """
## Files
- ``config.py`` — resolved training config
- ``summary.json`` — run summary (``tools/summarize_experiment``)
- ``metrics.json`` — key metrics for this release
- ``pytorch/best.pth`` — promoted weights
- ``onnx/`` — mmdeploy / ONNX Runtime bundle (if present)
- ``samples/`` — optional input / output demo images
"""
