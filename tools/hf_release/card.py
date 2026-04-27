# Copyright (c) OpenMMLab. Mowing-terrain-seg. SPDX-License-Identifier: Apache-2.0
"""Hugging Face model card (README) text from a release summary."""

from __future__ import annotations

import re
from typing import Any, Dict, List


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
- ``tensorrt/<profile>/`` — Jetson / TensorRT engine builds (optional)
- ``samples/`` — optional input / output demo images
"""


TENSORRT_SECTION = "## Available TensorRT engines"


def _parse_tensorrt_table_rows(section_body: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in section_body.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if re.match(r"^\|\s*-+", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 1 and cells[0].lower() in ("profile",):
            continue
        if cells and cells[0]:
            rows.append(cells)
    return rows


def _row_from_platform_json(platform: Dict[str, Any]) -> List[str]:
    prof = str(platform.get("profile", "unknown"))
    build = platform.get("build") or {}
    sw = platform.get("software") or {}
    return [
        prof,
        str(build.get("precision", "")),
        str(sw.get("tensorrt_python", "")),
        str(sw.get("cuda_cudart", "")),
        str(build.get("build_date", "")),
    ]


def _render_tensorrt_table(rows: List[List[str]]) -> str:
    header = (
        "| Profile | Precision | TRT | CUDA | Built |\n"
        "|---|---|---|---|---|"
    )
    body_lines: List[str] = []
    for r in rows:
        while len(r) < 5:
            r.append("")
        body_lines.append(
            "| " + " | ".join(r[:5]) + " |"
        )
    return TENSORRT_SECTION + "\n\n" + header + "\n" + "\n".join(body_lines) + "\n"


def merge_tensorrt_engines_readme(
    existing_readme: str, platform_json: Dict[str, Any]
) -> str:
    """Insert or update the ``## Available TensorRT engines`` section."""
    new_row = _row_from_platform_json(platform_json)
    new_profile = new_row[0]
    if TENSORRT_SECTION in existing_readme:
        start = existing_readme.index(TENSORRT_SECTION)
        rest = existing_readme[start + len(TENSORRT_SECTION) :]
        m = re.search(r"\n##\s+[^\n]", rest)
        if m:
            end = start + len(TENSORRT_SECTION) + m.start()
            after = existing_readme[end:]
        else:
            end = len(existing_readme)
            after = ""
        old_section = existing_readme[start:end]
        before = existing_readme[:start]
        old_body = old_section.split(TENSORRT_SECTION, 1)[-1]
        old_rows = _parse_tensorrt_table_rows(old_body)
        merged: List[List[str]] = []
        seen = False
        for r in old_rows:
            if r and r[0] == new_profile:
                merged.append(new_row)
                seen = True
            else:
                merged.append(r[:5] if len(r) >= 5 else r + [""] * (5 - len(r)))
        if not seen:
            merged.append(new_row)
        return before + _render_tensorrt_table(merged) + (after or "")
    return existing_readme.rstrip() + "\n\n" + _render_tensorrt_table([new_row]) + "\n"
