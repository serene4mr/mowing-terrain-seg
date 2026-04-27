# Copyright (c) OpenMMLab. Mowing-terrain-seg. SPDX-License-Identifier: Apache-2.0
"""Auto-detect Jetson / L4T environment for TensorRT engine ``platform.json`` metadata."""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

# board model substring -> (short slug, sm_capability string)
# SM for Jetson: Orin = 8.7, Xavier = 7.2, etc.
_JETSON_BOARD_MAP: List[tuple[str, str, str]] = [
    ("Jetson Orin NX", "orin-nx", "8.7"),
    ("Jetson Orin Nano", "orin-nano", "8.7"),
    ("Jetson Orin", "orin", "8.7"),
    ("Jetson AGX Orin", "orin", "8.7"),
    ("Xavier", "xavier", "7.2"),
    ("TX2", "tx2", "6.2"),
]


def _read_file(path, max_len: int = 4096) -> Optional[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(max_len).strip()
    except OSError:
        return None


def _read_l4t_string() -> Optional[str]:
    return _read_file("/etc/nv_tegra_release")


def _read_device_tree_model() -> Optional[str]:
    return _read_file("/proc/device-tree/model", max_len=256)


def _read_meminfo_gb() -> Optional[int]:
    raw = _read_file("/proc/meminfo", 512)
    if not raw:
        return None
    m = re.search(r"MemTotal:\s+(\d+)\s+kB", raw)
    if not m:
        return None
    kb = int(m.group(1))
    return int(round(kb / (1024.0**2)))


def _run_cmd(args: List[str], timeout: int = 10) -> Optional[str]:
    try:
        out = subprocess.check_output(
            args,
            text=True,
            timeout=timeout,
            stderr=subprocess.DEVNULL,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return None


def _nvpmodel_mode() -> Optional[str]:
    out = _run_cmd(["/usr/sbin/nvpmodel", "-q"])
    if not out and sys.platform == "linux":
        out = _run_cmd(["nvpmodel", "-q"])
    if not out:
        return None
    m = re.search(r"(\d+W|MAXN|model\s*(\d+))", out, re.I)
    if m:
        return m.group(0).replace(" ", "")
    line = out.splitlines()[0] if out else out
    return line[:64]


def _apt_show_nvidia_jetpack() -> Optional[str]:
    out = _run_cmd(["/usr/bin/apt", "show", "nvidia-jetpack", "-a"])
    if not out:
        return None
    m = re.search(r"^Version:\s*(\S+)", out, re.M)
    return m.group(1) if m else None


def _dpkg_grep_version(pattern: str) -> Optional[str]:
    out = _run_cmd(["/usr/bin/dpkg", "-l"])
    if not out:
        return None
    for line in out.splitlines():
        if re.search(pattern, line, re.I):
            parts = line.split()
            if len(parts) >= 3:
                return parts[2]
    return None


def _import_tensorrt_version() -> Optional[str]:
    try:
        import tensorrt as trt  # type: ignore[import-not-found]
    except ImportError:
        return None
    v = getattr(trt, "__version__", None)
    return str(v) if v is not None else None


def _board_to_slug_and_sm(model: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    if not model:
        return None, None, None
    m = model.strip().strip("\x00")
    for needle, slug, sm in _JETSON_BOARD_MAP:
        if needle.lower() in m.lower():
            return m, slug, sm
    # Unknown Jetson: slugify a short form
    slug = re.sub(r"[^a-z0-9]+", "-", m.lower())[:40].strip("-")
    return m, slug or "jetson", None


def _l4t_major_minor(l4t: Optional[str]) -> Optional[str]:
    if not l4t:
        return None
    m = re.search(r"R(\d+)\s*\(release\)\s*,\s*REVISION:\s*([\d.]+)", l4t, re.I)
    if m:
        return f"r{m.group(1)}.{m.group(2)}"
    m2 = re.search(r"R(\d+)[.\s]([\d.]+)", l4t)
    if m2:
        return f"r{m2.group(1)}.{m2.group(2)}"
    return None


def _jetpack_guess(l4t_mm: Optional[str], jetpack: Optional[str]) -> Optional[str]:
    if jetpack:
        v = re.sub(r"\+b\d+$", "", jetpack)  # 6.2.2+b24 -> 6.2.2
        return v
    if l4t_mm and l4t_mm.startswith("r36.5"):
        return "6.2"
    if l4t_mm and l4t_mm.startswith("r36.4"):
        return "6.1"
    return None


def detect_jetson_profile() -> Dict[str, Any]:
    """Best-effort read of the local environment. Missing fields are omitted or null."""
    l4t = _read_l4t_string()
    model, board_slug, sm = _board_to_slug_and_sm(
        _read_device_tree_model()
    )
    mem_g = _read_meminfo_gb()
    l4t_mm = _l4t_major_minor(l4t)
    jp = _apt_show_nvidia_jetpack()
    jpg = _jetpack_guess(l4t_mm, jp)

    out: Dict[str, Any] = {
        "l4t_raw": l4t,
        "l4t": l4t_mm,
        "board_model": model,
        "board_slug": board_slug,
        "memory_gb": mem_g,
        "sm_capability": sm,
        "jetpack_apt": jp,
        "jetpack": jpg,
        "power_mode": _nvpmodel_mode(),
        "cuda_cudart": _dpkg_grep_version(r"^ii\s+cuda-cudart-12"),
        "cudnn": _dpkg_grep_version(r"^ii\s+libcudnn9-cuda-12"),
        "tensorrt_python": _import_tensorrt_version(),
    }
    return {k: v for k, v in out.items() if v is not None}


def build_profile_name(detected: Dict[str, Any], precision: str) -> str:
    """Build a filesystem-safe profile directory name, e.g. orin-nx-16gb-jp6.2-fp16-trt10.4."""
    board = detected.get("board_slug") or "jetson"
    mem = detected.get("memory_gb")
    mem_s = f"{int(mem)}gb" if mem is not None else "unknown"
    jp = str(detected.get("jetpack") or detected.get("l4t") or "unknown")
    jp_s = re.sub(r"^r", "", str(jp).replace(" ", ""))[:12]
    pr = (precision or "fp16").lower().strip()
    trt = detected.get("tensorrt_python")
    if trt:
        trt_s = str(trt).split(".", 2)
        trt_lbl = f"trt{trt_s[0]}.{trt_s[1]}" if len(trt_s) >= 2 else f"trt{trt}"
    else:
        trt_lbl = "trt-unknown"
    return f"{board}-{mem_s}-jp{jp_s}-{pr}-{trt_lbl}"


def render_platform_json(
    *,
    profile: str,
    detected: Dict[str, Any],
    build: Dict[str, Any],
    source: Dict[str, Any],
    docker_image: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Assemble the ``platform.json`` object written by ``build_engine``."""
    out: Dict[str, Any] = {
        "schema_version": "1.0",
        "profile": profile,
    }
    device: Dict[str, Any] = {}
    for k in ("board_model", "memory_gb", "sm_capability", "power_mode"):
        if k in detected and detected[k] is not None:
            device[k] = detected[k]
    if device:
        out["device"] = device
    software: Dict[str, Any] = {}
    for k in ("l4t", "l4t_raw", "jetpack", "jetpack_apt", "cuda_cudart", "cudnn", "tensorrt_python"):
        if k in detected and detected[k] is not None:
            software[k] = detected[k]
    if software:
        out["software"] = software
    if docker_image:
        out["image"] = docker_image
    out["build"] = dict(build)
    out["source"] = dict(source)
    return out
