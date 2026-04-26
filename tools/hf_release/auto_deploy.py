# Copyright (c) OpenMMLab. Mowing-terrain-seg. SPDX-License-Identifier: Apache-2.0
"""Re-run `tools/deploy/deploy.py` when ONNX is missing or older than a checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path
from subprocess import run as subprocess_run
from typing import List


def should_redeploy(deploy_dir: Path, pth: Path) -> bool:
    """True if we should re-export to ONNX (missing dir, missing onnx, or pth is newer)."""
    d = deploy_dir
    pth = pth.resolve()
    if not d.exists():
        return True
    onnx = d / "end2end.onnx"
    if not onnx.is_file():
        return True
    if pth.stat().st_mtime > onnx.stat().st_mtime + 0.1:
        return True
    return False


def run_deploy(
    *,
    deploy_py: Path,
    deploy_cfg: Path,
    model_cfg: Path,
    pth: Path,
    sample_img: Path,
    work_dir: Path,
    device: str = "cpu",
    extra_args: List[str] | None = None,
) -> None:
    """Call ``tools/deploy/deploy.py`` with the same contract as the CLI (positional 4 + options)."""
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        sys.executable,
        str(deploy_py.resolve()),
        str(deploy_cfg.resolve()),
        str(model_cfg.resolve()),
        str(pth.resolve()),
        str(sample_img.resolve()),
        "--work-dir",
        str(work_dir),
        "--device",
        device,
        "--dump-info",
    ]
    if extra_args:
        cmd.extend(extra_args)
    subprocess_run(
        cmd,
        check=True,
    )
