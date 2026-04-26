#!/usr/bin/env python3
"""
Verify that the mowing-terrain-seg dev environment satisfies all requirements.

Usage:
    python3 docker/verify_env.py
    python3 docker/verify_env.py --model work_dirs/deploy2   # custom model dir
    python3 docker/verify_env.py --no-inference              # skip GPU inference
"""
import argparse
import glob
import os
import sys

# ── ANSI colours ─────────────────────────────────────────────────────────────
GREEN = "\033[32m"
RED   = "\033[31m"
RESET = "\033[0m"
PASS  = f"{GREEN}[PASS]{RESET}"
FAIL  = f"{RED}[FAIL]{RESET}"
W     = 42


def check(label: str, display: str, ok: bool) -> bool:
    tag = PASS if ok else FAIL
    print(f"  {label:<{W}} {display}  {tag}")
    return ok


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="work_dirs/deploy2",
                   help="Path to mmdeploy model directory (default: work_dirs/deploy2)")
    p.add_argument("--no-inference", action="store_true",
                   help="Skip the GPU inference test")
    return p.parse_args()


def main():
    args = parse_args()
    results = []

    print("\n── Environment verification ─────────────────────────────────────────")

    # ── 1. Python ─────────────────────────────────────────────────────────────
    v = sys.version.split()[0]
    results.append(check("1. Python", v, v.startswith("3.10")))

    # ── 2. PyTorch + CUDA build ───────────────────────────────────────────────
    import torch
    ok = torch.__version__ == "2.1.2" and torch.version.cuda == "11.8"
    results.append(check("2. PyTorch", f"{torch.__version__}+cu{torch.version.cuda}", ok))

    # ── 3. GPU visible ────────────────────────────────────────────────────────
    avail = torch.cuda.is_available()
    name  = torch.cuda.get_device_name(0) if avail else "no CUDA device"
    results.append(check("3. GPU", name, avail))

    # ── 4. OpenMMLab ──────────────────────────────────────────────────────────
    try:
        import mmcv, mmdet, mmseg  # mmsegmentation installs as 'mmseg'
        for mod, exp in [(mmcv, "2.1.0"), (mmdet, "3.3.0"), (mmseg, "1.2.2")]:
            ok = mod.__version__ == exp
            results.append(check(f"4. {mod.__name__}", mod.__version__, ok))
    except ImportError as e:
        results.append(check("4. OpenMMLab", str(e), False))

    try:
        import mowing_terrain_seg

        mowing_terrain_seg.register_all()
        results.append(
            check("4b. mowing_terrain_seg", "import + register_all", True)
        )
    except Exception as e:  # noqa: BLE001
        results.append(check("4b. mowing_terrain_seg", str(e), False))

    # ── 5. ORT C++ SDK tarball ────────────────────────────────────────────────
    ort_libs = glob.glob("/opt/onnxruntime/lib/libonnxruntime*.so*")
    has_cuda   = any("providers_cuda"   in l for l in ort_libs)
    has_shared = any("providers_shared" in l for l in ort_libs)
    results.append(check("5. ORT tarball providers_cuda", "/opt/onnxruntime/lib",
                         has_cuda and has_shared))

    # ── 6. Environment variables ──────────────────────────────────────────────
    ort_dir = os.environ.get("ONNXRUNTIME_DIR", "")
    ld      = os.environ.get("LD_LIBRARY_PATH", "")
    results.append(check("6. ONNXRUNTIME_DIR", ort_dir, ort_dir == "/opt/onnxruntime"))
    results.append(check("6. LD_LIBRARY_PATH ⊇ /opt/onnxruntime/lib",
                         str("/opt/onnxruntime/lib" in ld),
                         "/opt/onnxruntime/lib" in ld))

    # ── 7. MMDeploy packages ──────────────────────────────────────────────────
    try:
        import mmdeploy
        import mmdeploy_runtime as mmd  # noqa: F401 (also validates shared-lib load)
        results.append(check("7. mmdeploy", mmdeploy.__version__,
                             mmdeploy.__version__ == "1.3.1"))
        results.append(check("7. mmdeploy_runtime", mmd.__version__,
                             mmd.__version__ == "1.3.1"))
    except Exception as e:
        results.append(check("7. mmdeploy_runtime", str(e), False))
        mmd = None

    # ── 8. GPU inference ──────────────────────────────────────────────────────
    if args.no_inference:
        print(f"  {'8. GPU inference':<{W}} skipped (--no-inference)")
    else:
        try:
            import numpy as np
            import time

            model_dir = args.model
            if not os.path.isdir(model_dir):
                results.append(check("8. model dir exists", model_dir, False))
            else:
                seg = mmd.Segmentor(model_dir, "cuda", 0)
                img = np.random.randint(0, 255, (544, 1024, 3), dtype=np.uint8)
                _ = seg(img)                      # warm-up
                t0 = time.perf_counter()
                for _ in range(3):
                    out = seg(img)
                ms = (time.perf_counter() - t0) / 3 * 1000
                mask = out.argmax(axis=0)
                classes = sorted(set(mask.flatten().tolist()))
                results.append(check("8. GPU inference shape",
                                     str(out.shape), out.shape == (3, 544, 1024)))
                results.append(check("8. mask classes", str(classes), True))
                results.append(check("8. GPU latency", f"{ms:.0f} ms/frame", True))
        except Exception as e:
            results.append(check("8. GPU inference", str(e), False))

    # ── Summary ───────────────────────────────────────────────────────────────
    total   = len(results)
    passing = sum(results)
    print()
    if passing == total:
        print(f"{GREEN}  ✓ All {total} checks passed.{RESET}")
    else:
        failed = total - passing
        print(f"{RED}  ✗ {failed}/{total} checks failed.{RESET}")
        sys.exit(1)
    print()


if __name__ == "__main__":
    main()
