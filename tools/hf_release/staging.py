# Copyright (c) OpenMMLab. Mowing-terrain-seg. SPDX-License-Identifier: Apache-2.0
"""Build a local folder matching the Hugging Face Hub model repo layout."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tools.summarize_experiment import find_root_config_py

# Default ONNX / SDK files to copy; also copy any other ``*.onnx*`` in the dir.
_ONNX_BUNDLE: Tuple[str, ...] = (
    "end2end.onnx",
    "pipeline.json",
    "deploy.json",
    "detail.json",
)


def _copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_staging(
    *,
    exp_dir: Path,
    pth: Path,
    onnx_dir: Optional[Path],
    summary: Dict[str, Any],
    sample_in: Optional[Path],
    sample_out: Optional[Path],
    dest: Path,
) -> None:
    """Populate ``dest`` with config, ``summary.json``, ``pytorch/best.pth``, ``onnx/``, ``samples/``."""
    exp_dir = exp_dir.resolve()
    dest = dest.resolve()
    if dest.is_file():
        raise ValueError(f"dest must be a directory, got: {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    cfg = find_root_config_py(exp_dir)
    if cfg is None or not cfg.is_file():
        raise OSError(
            "No top-level training config .py in exp_dir; run validate_experiment first."
        )
    _copy_file(cfg, dest / "config.py")

    with open(dest / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
            sort_keys=True,
        )

    pth = pth.resolve()
    _copy_file(pth, dest / "pytorch" / "best.pth")

    if onnx_dir is not None:
        d = onnx_dir.resolve()
        copied: set[str] = set()
        for name in _ONNX_BUNDLE:
            p = d / name
            if p.is_file():
                _copy_file(p, dest / "onnx" / name)
                copied.add(name)
        if d.is_dir():
            for p in d.iterdir():
                if (
                    p.is_file()
                    and p.suffix in {".onnx", ".json", ".onnxdata"}
                    and p.name not in copied
                ):
                    _copy_file(p, dest / "onnx" / p.name)
                    copied.add(p.name)

    if sample_in is not None and sample_out is not None:
        if sample_in.is_file() and sample_out.is_file():
            _copy_file(sample_in, dest / "samples" / "input.jpg")
            _copy_file(sample_out, dest / "samples" / "output.png")
