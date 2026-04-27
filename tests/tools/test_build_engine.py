# Copyright (c) OpenMMLab. Mowing-terrain-seg.
"""Tests for tools/build_engine.py (TensorRT build orchestration, mocked)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture
def local_onnx(tmp_path: Path) -> Path:
    p = tmp_path / "end2end.onnx"
    p.write_bytes(b"x" * 32)
    return p


def test_dry_run_no_trtexec(
    tmp_path: Path, local_onnx: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.build_engine as be

    monkeypatch.setattr(
        "tools.build_engine.jetson_platform.detect_jetson_profile",
        mock.Mock(
            return_value={"board_slug": "orin-nx", "memory_gb": 16, "jetpack": "6.2"}
        ),
    )
    r = be.main(
        [
            "--no-pull",
            "--onnx",
            str(local_onnx),
            "--output-dir",
            str(tmp_path / "o"),
            "--dry-run",
        ]
    )
    assert r == 0


def test_trtexec_invocation_fp16(
    tmp_path: Path, local_onnx: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.build_engine as be

    out = tmp_path / "b"
    out.mkdir()
    trt = tmp_path / "trtexec"
    trt.write_text("#!/bin/sh\necho ok > \"$5\"\n", encoding="utf-8")
    trt.chmod(0o755)
    # Use shell-style mock: the real trtexec is not available in CI
    runs = {"n": 0}

    class _FakePopen:  # noqa: D101
        def __init__(self, cmd, **kwargs):  # type: ignore[no-untyped-def]
            self.cmd = cmd
            self.returncode = 0
            self.stdout = iter(["ok\n"])
            runs["n"] += 1
            eng = None
            for a in cmd:
                if a.startswith("--saveEngine="):
                    eng = Path(a.split("=", 1)[1])
            assert eng is not None
            eng.parent.mkdir(parents=True, exist_ok=True)
            eng.write_bytes(b"TRT1")

        def wait(self):  # type: ignore[no-untyped-def]
            return self.returncode

    def fake_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        return _FakePopen(cmd, **kwargs)

    monkeypatch.setattr(be.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(be, "_default_trtexec", lambda: trt)  # path exists; still mock run
    # point trtexec in cmd to our fake: fake_run is used so actual binary unused
    monkeypatch.setattr(
        "tools.build_engine.jetson_platform.detect_jetson_profile",
        mock.Mock(
            return_value={"board_slug": "orin-nx", "memory_gb": 8, "tensorrt_python": "10.0.0"}
        ),
    )
    r = be.main(
        [
            "--no-pull",
            "--onnx",
            str(local_onnx),
            "--output-dir",
            str(out),
            "--precision",
            "fp16",
        ]
    )
    assert r == 0
    assert (out / "end2end.engine").is_file()
    pl = json.loads((out / "platform.json").read_text(encoding="utf-8"))
    assert pl["source"]["onnx_sha256"]
    assert pl["build"]["precision"] == "fp16"
    assert "fp16" in pl["build"]["trtexec_args"] or any(
        "fp16" in str(x) for x in pl["build"].get("trtexec_args", [])
    )
    assert runs["n"] == 1
