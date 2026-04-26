"""
Replace mmdeploy/mmcv custom ops with standard ONNX ops where possible.

Supported replacements:
- mmdeploy::grid_sampler -> GridSample (opset 13+)
- mmcv::MMCVRoIAlign -> RoiAlign (opset 16+)

Other custom ops (mmdeploy/mmcv) are left as-is and reported.

Run: python tools/deploy/rewrite_custom_ops_onnx.py \\
  mmdeploy_model/mask2former-onnx/end2end.onnx \\
  mmdeploy_model/mask2former-onnx/end2end_standard.onnx
"""
import argparse
import os
import sys

_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEPLOY_DIR not in sys.path:
    sys.path.insert(0, _DEPLOY_DIR)
from _onnx_rewriter import rewrite_custom_ops_to_file  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description="Replace mmdeploy/mmcv custom ops with standard ONNX where possible."
    )
    p.add_argument("input_onnx", help="Path to input ONNX (e.g. end2end.onnx)")
    p.add_argument("output_onnx", help="Path for rewritten ONNX")
    p.add_argument(
        "--allow-custom-ops",
        action="store_true",
        help="Exit 0 even if custom ops remain (default: exit 2 if any remain).",
    )
    args = p.parse_args()
    _, kept = rewrite_custom_ops_to_file(args.input_onnx, args.output_onnx)
    print(f"Saved to {args.output_onnx}")
    print(f"Replaced custom op(s); kept_custom: {kept}")
    if kept and not args.allow_custom_ops:
        print(
            "Custom mmdeploy/mmcv ops remain. Use --allow-custom-ops to exit 0.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
