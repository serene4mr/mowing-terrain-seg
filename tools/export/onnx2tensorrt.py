"""Build a TensorRT engine from a local ONNX file using trtexec."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence


def _default_trtexec() -> Path:
    env = os.environ.get("TRTEXEC")
    if env:
        return Path(env)
    # Common locations for trtexec on Jetson and desktop
    candidates = [
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return Path(c)
    return Path("trtexec")  # Fallback to PATH


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a local ONNX model to a TensorRT engine."
    )
    p.add_argument(
        "--onnx",
        type=Path,
        required=True,
        help="Path to the input .onnx file.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work_dirs/export_tensorrt"),
        help="Directory to save the built .engine file and logs.",
    )
    p.add_argument(
        "--precision",
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="Target inference precision (default: fp16).",
    )
    p.add_argument(
        "--workspace-mb",
        type=int,
        default=2048,
        help="Max workspace size for TensorRT builder in MB.",
    )
    p.add_argument(
        "--input-shape",
        default="1,3,512,512",
        help="Input shape as comma-separated integers (e.g. 1,3,512,512).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print trtexec command and exit without running.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    
    onnx_path = args.onnx.resolve()
    if not onnx_path.is_file():
        print(f"Error: ONNX file not found at {onnx_path}", file=sys.stderr)
        return 1

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    engine_name = f"model_{args.precision}.engine"
    engine_out = out_dir / engine_name
    
    trtexec = _default_trtexec()

    # Build extra TRT flags
    extra_trt: List[str] = []
    if args.precision == "fp16":
        extra_trt.append("--fp16")
    elif args.precision == "int8":
        extra_trt.append("--int8")

    # Parse shape and format for trtexec
    shape_parts = [x.strip() for x in args.input_shape.split(",") if x.strip()]
    shapes_arg = f"input:{'x'.join(shape_parts)}"

    cmd: List[str] = [
        str(trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_out}",
        f"--memPoolSize=workspace:{args.workspace_mb}M",
        f"--shapes={shapes_arg}",
    ] + extra_trt

    if args.dry_run:
        print("Would run:", " ".join(shlex.quote(c) for c in cmd))
        return 0

    print(f"Starting TensorRT conversion: {onnx_path.name} -> {engine_name}")
    print("Command:", " ".join(shlex.quote(c) for c in cmd))
    print("-" * 60)

    t0 = time.perf_counter()
    log_path = out_dir / f"{onnx_path.stem}_build.log"
    
    # Stream trtexec output live to terminal and persist to build.log
    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        pop = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert pop.stdout is not None
        for line in pop.stdout:
            sys.stdout.write(line)
            logf.write(line)
        pop.wait()
        rc = int(pop.returncode or 0)
        
    elapsed = time.perf_counter() - t0
    
    print("-" * 60)
    if rc != 0:
        print(f"Error: trtexec failed with code {rc}", file=sys.stderr)
        print(f"See full log: {log_path}", file=sys.stderr)
        return rc
        
    if not engine_out.is_file():
        print(f"Error: Expected engine at {engine_out} but file missing.", file=sys.stderr)
        return 1

    print(f"Successfully built TensorRT engine in {elapsed:.1f} seconds.")
    print(f"Engine saved to: {engine_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
