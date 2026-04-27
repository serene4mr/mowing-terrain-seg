# Copyright (c) OpenMMLab. Mowing-terrain-seg. SPDX-License-Identifier: Apache-2.0
"""Stage TensorRT engine artifacts for Hugging Face Hub upload."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Tuple


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def build_engine_staging(
    *,
    engine_dir: Path,
    dest: Path,
    profile: str,
) -> Tuple[Path, Path]:
    """Copy engine bundle into ``dest/tensorrt/<profile>/``.

    Expects ``engine_dir`` to contain at least ``end2end.engine`` and
    ``platform.json``. Optionally copies ``build.log``.

    Returns ``(tensorrt_root, profile_dir)`` where profile_dir is
    ``dest/tensorrt/<profile>``.
    """
    engine_dir = engine_dir.resolve()
    dest = dest.resolve()
    if dest.is_file():
        raise ValueError(f"dest must be a directory, got: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    prof = profile.strip().replace("..", "").strip("/") or "unknown"
    root = dest / "tensorrt"
    pdir = root / prof
    pdir.mkdir(parents=True, exist_ok=True)

    eng = engine_dir / "end2end.engine"
    plat = engine_dir / "platform.json"
    logf = engine_dir / "build.log"
    if not eng.is_file():
        raise OSError(f"Missing {eng}")
    if not plat.is_file():
        raise OSError(f"Missing {plat}")
    shutil.copy2(eng, pdir / "end2end.engine")
    shutil.copy2(plat, pdir / "platform.json")
    _copy_if_exists(logf, pdir / "build.log")
    return root, pdir
