# Copyright (c) OpenMMLab. Mowing-terrain-seg. SPDX-License-Identifier: Apache-2.0
"""Build a TensorRT engine from ONNX (pulled from Hugging Face Hub or local path)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.hf_release import platform as jetson_platform  # noqa: E402


def _sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _default_trtexec() -> Path:
    env = os.environ.get("TRTEXEC")
    if env:
        return Path(env)
    return Path("/usr/src/tensorrt/bin/trtexec")


def _parse_shape(s: str) -> Tuple[int, ...]:
    parts = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(parts) != 4:
        raise ValueError(f"Expected 4 comma-separated ints (N,C,H,W), got: {s!r}")
    return tuple(parts)


def _pull_snapshot(
    repo_id: str,
    revision: str,
    dest: Path,
    token: Optional[str],
) -> Path:
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub is not installed. Install: pip install -e '.[release]'"
        ) from e
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    local = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        allow_patterns=["onnx/*", "summary.json", "config.py"],
        token=token,
    )
    return Path(local)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build TensorRT engine from ONNX (Hub pull or local onnx path)."
    )
    p.add_argument(
        "--repo-id",
        help="HF model repo id (e.g. org/model). Required unless --onnx is set.",
    )
    p.add_argument(
        "--revision",
        default="main",
        help="HF revision (branch, tag, or commit) for snapshot download.",
    )
    p.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="Local path to end2end.onnx (skip Hub pull when set with --no-pull).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write end2end.engine, platform.json, build.log, source/.",
    )
    p.add_argument(
        "--precision",
        choices=("fp32", "fp16", "int8"),
        default="fp16",
        help="TensorRT precision flag (default fp16).",
    )
    p.add_argument(
        "--workspace-mb",
        type=int,
        default=4096,
        help="TensorRT workspace pool size in MiB (default 4096).",
    )
    p.add_argument(
        "--input-shape",
        type=str,
        default="1,3,1024,544",
        help="Static input shape N,C,H,W for trtexec --shapes (comma-separated).",
    )
    p.add_argument(
        "--profile",
        default=None,
        help="Override auto-detected profile directory name.",
    )
    p.add_argument(
        "--no-pull",
        action="store_true",
        help="Use --onnx file directly; do not call snapshot_download.",
    )
    p.add_argument(
        "--docker-image-ref",
        default=None,
        help="Optional image ref (e.g. ghcr.io/...:tag) recorded in platform.json.",
    )
    p.add_argument(
        "--docker-image-digest",
        default=None,
        help="Optional image digest sha256:... for platform.json.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan and exit without running trtexec.",
    )
    p.add_argument(
        "--token",
        default=None,
        help="HF token (or set HF_TOKEN env).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = out_dir / "source"
    src_dir.mkdir(parents=True, exist_ok=True)

    onnx_path: Optional[Path] = None
    repo_id = args.repo_id
    revision = args.revision

    if args.no_pull:
        if args.onnx is None or not args.onnx.is_file():
            print(
                "--no-pull requires --onnx pointing to an existing end2end.onnx",
                file=sys.stderr,
            )
            return 2
        src_onnx = args.onnx.resolve()
        shutil.copy2(src_onnx, src_dir / "end2end.onnx")
        onnx_path = src_dir / "end2end.onnx"
        if repo_id is None:
            repo_id = "local"
    else:
        if not repo_id:
            print("--repo-id is required when not using --no-pull", file=sys.stderr)
            return 2
        snap_root = out_dir / "_hf_snapshot"
        if snap_root.is_dir():
            shutil.rmtree(snap_root)
        print(f"Downloading snapshot {repo_id}@{revision} ...")
        local_root = _pull_snapshot(
            repo_id, revision, snap_root, token=args.token or os.environ.get("HF_TOKEN")
        )
        cand = local_root / "onnx" / "end2end.onnx"
        if not cand.is_file():
            print(f"Missing {cand} after download.", file=sys.stderr)
            return 2
        shutil.copy2(cand, src_dir / "end2end.onnx")
        for name in ("summary.json", "config.py"):
            p = local_root / name
            if p.is_file():
                shutil.copy2(p, src_dir / name)
        onnx_path = src_dir / "end2end.onnx"

    assert onnx_path is not None
    onnx_sha = _sha256_file(onnx_path)

    detected = jetson_platform.detect_jetson_profile()
    precision = args.precision
    profile = args.profile or jetson_platform.build_profile_name(detected, precision)

    shape = _parse_shape(args.input_shape)
    engine_out = out_dir / "end2end.engine"
    trtexec = _default_trtexec()
    if not trtexec.is_file() and not args.dry_run:
        print(
            f"trtexec not found at {trtexec}. Set TRTEXEC or install TensorRT.",
            file=sys.stderr,
        )
        return 2

    extra_trt: List[str] = []
    if precision == "fp16":
        extra_trt.append("--fp16")
    elif precision == "int8":
        extra_trt.append("--int8")
    # fp32: no extra flag

    # TensorRT trtexec expects dimensions in NxCxHxW with "x" separators,
    # e.g. --shapes=input:1x3x1024x544 (commas are parsed as multi-input separators).
    shapes_arg = f"input:{'x'.join(str(x) for x in shape)}"
    cmd: List[str] = [
        str(trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_out}",
        f"--memPoolSize=workspace:{args.workspace_mb}M",
        f"--shapes={shapes_arg}",
    ] + extra_trt

    if args.dry_run:
        print("Profile:", profile)
        print("Would run:", " ".join(shlex.quote(c) for c in cmd))
        print("ONNX sha256:", onnx_sha)
        return 0

    t0 = time.perf_counter()
    log_path = out_dir / "build.log"
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    log_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(f"trtexec failed with code {proc.returncode}", file=sys.stderr)
        return proc.returncode or 1
    if not engine_out.is_file():
        print(f"Expected engine at {engine_out} but file missing.", file=sys.stderr)
        return 1

    build_date = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    build_meta: Dict[str, Any] = {
        "precision": precision,
        "workspace_mb": args.workspace_mb,
        "input_shape": list(shape),
        "trtexec": str(trtexec),
        "trtexec_args": cmd[1:],
        "duration_sec": round(elapsed, 3),
        "build_date": build_date,
    }
    source_meta: Dict[str, Any] = {
        "onnx_repo": repo_id,
        "onnx_revision": revision,
        "onnx_sha256": onnx_sha,
        "onnx_path": "onnx/end2end.onnx",
    }
    docker_image: Optional[Dict[str, str]] = None
    if args.docker_image_ref or args.docker_image_digest:
        docker_image = {}
        if args.docker_image_ref:
            docker_image["ref"] = args.docker_image_ref
        if args.docker_image_digest:
            docker_image["digest"] = args.docker_image_digest

    plat = jetson_platform.render_platform_json(
        profile=profile,
        detected=detected,
        build=build_meta,
        source=source_meta,
        docker_image=docker_image,
    )
    (out_dir / "platform.json").write_text(
        json.dumps(plat, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {engine_out}")
    print(f"Wrote {out_dir / 'platform.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
