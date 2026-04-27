# Copyright (c) OpenMMLab. Mowing-terrain-seg.
"""Tests for tools/hf_release/platform.py (Jetson profile detection and naming)."""

from __future__ import annotations

from tools.hf_release import platform as p


def test_board_to_sm_orin_nx() -> None:
    m, slug, sm = p._board_to_slug_and_sm("NVIDIA Jetson Orin NX module")
    assert "Orin" in (m or "")
    assert slug == "orin-nx"
    assert sm == "8.7"


def test_build_profile_name() -> None:
    d = {
        "board_slug": "orin-nx",
        "memory_gb": 16,
        "jetpack": "6.2.2",
        "tensorrt_python": "10.4.0",
    }
    name = p.build_profile_name(d, "fp16")
    assert "orin-nx" in name
    assert "16gb" in name
    assert "fp16" in name
    assert "trt10.4" in name


def test_jetpack_guess() -> None:
    assert p._jetpack_guess("r36.5.0", "6.2.2+b24") == "6.2.2"
    assert p._jetpack_guess("r36.4.0", None) == "6.1"
    assert p._jetpack_guess("r36.5.0", None) == "6.2"


def test_render_platform_json_minimal() -> None:
    out = p.render_platform_json(
        profile="p1",
        detected={"board_model": "X"},
        build={"precision": "fp16"},
        source={"onnx_sha256": "a" * 64, "onnx_repo": "o/m", "onnx_revision": "v1"},
    )
    assert out["schema_version"] == "1.0"
    assert out["profile"] == "p1"
    assert out["source"]["onnx_sha256"] == "a" * 64
