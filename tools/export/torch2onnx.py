"""
Export PyTorch model to ONNX using mmdeploy configs, with optional custom-op rewrite.

Flow:
- Load deploy/model configs and validate that IR type is ONNX.
- Export checkpoint to ONNX via mmdeploy torch2onnx using the provided sample image.
- If partition config exists, split the exported graph into partition ONNX files.
- If calibration output is configured, generate calibration input data.
- Unless disabled, rewrite supported mmdeploy/mmcv custom ops to standard ONNX ops in place.
- Report exported ONNX artifact paths and fail with exit code 2 when custom ops remain
  (unless --allow-custom-ops is set).

Usage:
  python tools/export/torch2onnx.py <deploy_cfg> <model_cfg> <checkpoint> <img> \
    --work-dir work_dirs/export_onnx --device cuda

  # Generate SDK information (pipeline.json, etc.)
  python tools/export/torch2onnx.py ... --dump-info

  # Keep custom ops (do not fail if any remain after rewrite)
  python tools/export/torch2onnx.py ... --allow-custom-ops

  # Skip rewrite step entirely
  python tools/export/torch2onnx.py ... --no-rewrite
"""

import argparse
import logging
import os
import os.path as osp
import sys

import mmengine
import onnx

from mmdeploy.apis import create_calib_input_data, extract_model, get_predefined_partition_cfg, torch2onnx
from mmdeploy.backend.sdk.export_info import export2SDK
from mmdeploy.utils import IR, get_calib_filename, get_ir_config, get_partition_config, get_root_logger, load_config

_EXPORT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_EXPORT_DIR))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
for _path in (_SRC_ROOT, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

_HELPERS_DIR = os.path.join(_EXPORT_DIR, "helpers")
if _HELPERS_DIR not in sys.path:
    sys.path.insert(0, _HELPERS_DIR)
from _onnx_rewriter import rewrite_model_in_memory  # noqa: E402  # pyright: ignore[reportMissingImports]


def parse_args():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX using deploy config.")
    parser.add_argument("deploy_cfg", help="deploy config path")
    parser.add_argument("model_cfg", help="model config path")
    parser.add_argument("checkpoint", help="model checkpoint path")
    parser.add_argument("img", help="image used to convert model")
    parser.add_argument(
        "--work-dir",
        default=os.getcwd(),
        help="the dir to save logs and models",
    )
    parser.add_argument(
        "--calib-dataset-cfg",
        default=None,
        help=(
            "dataset config path used to calibrate in int8 mode. "
            'If not specified, uses "val" dataset in model config.'
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="device used for conversion",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=list(logging._nameToLevel.keys()),
        help="set log level",
    )
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Skip rewriting custom ops to standard ONNX.",
    )
    parser.add_argument(
        "--allow-custom-ops",
        action="store_true",
        help="Do not exit with error if custom mmdeploy/mmcv ops remain after rewrite.",
    )
    parser.add_argument(
        "--dump-info",
        action="store_true",
        help="Output information for SDK.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print export plan without executing.",
    )
    return parser.parse_args()


def _resolve_partition_cfgs(deploy_cfg):
    partition_cfgs = get_partition_config(deploy_cfg)
    if partition_cfgs is None:
        return None

    if "partition_cfg" in partition_cfgs:
        return partition_cfgs.get("partition_cfg", None)

    assert "type" in partition_cfgs
    return get_predefined_partition_cfg(deploy_cfg, partition_cfgs["type"])


def _rewrite_onnx_files(onnx_files, logger, allow_custom_ops):
    any_kept = False

    for onnx_path in onnx_files:
        if not (onnx_path.endswith(".onnx") and osp.isfile(onnx_path)):
            continue

        model = onnx.load(onnx_path)
        model, replaced_count, kept_custom = rewrite_model_in_memory(model, logger=logger)
        onnx.save(model, onnx_path)
        logger.info(
            "Rewritten (standard ops) in place: %s, replaced %d custom op(s).",
            onnx_path,
            replaced_count,
        )
        if kept_custom:
            any_kept = True
            for (dom, op), count in sorted(kept_custom.items()):
                logger.info("  Kept %s::%s: %d node(s)", dom, op, count)

    if any_kept and not allow_custom_ops:
        logger.error(
            "Custom mmdeploy/mmcv ops remain after rewrite. "
            "Re-run with --allow-custom-ops to allow, or fix the model / deploy config."
        )
        raise SystemExit(2)


def main():
    args = parse_args()
    logger = get_root_logger()
    logger.setLevel(logging._nameToLevel[args.log_level])

    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg
    checkpoint_path = args.checkpoint

    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)
    mmengine.mkdir_or_exist(osp.abspath(args.work_dir))

    if args.dump_info:
        export2SDK(
            deploy_cfg,
            model_cfg,
            args.work_dir,
            pth=checkpoint_path,
            device=args.device,
        )

    if args.dry_run:
        logger.info("Dry run: skipping export.")
        return

    ir_config = get_ir_config(deploy_cfg)
    ir_type = IR.get(ir_config["type"])
    if ir_type != IR.ONNX:
        raise ValueError(f"Expected ONNX IR in deploy config, but got: {ir_type}")
    ir_save_file = ir_config["save_file"]

    logger.info("Exporting PyTorch checkpoint to ONNX...")
    torch2onnx(
        args.img,
        args.work_dir,
        ir_save_file,
        deploy_cfg_path,
        model_cfg_path,
        checkpoint_path,
        device=args.device,
    )
    logger.info("Primary ONNX export completed.")

    ir_files = [osp.join(args.work_dir, ir_save_file)]
    partition_cfgs = _resolve_partition_cfgs(deploy_cfg)
    if partition_cfgs is not None:
        origin_ir_file = ir_files[0]
        ir_files = []
        for partition_cfg in partition_cfgs:
            save_file = partition_cfg["save_file"]
            save_path = osp.join(args.work_dir, save_file)
            extract_model(
                origin_ir_file,
                partition_cfg["start"],
                partition_cfg["end"],
                dynamic_axes=partition_cfg.get("dynamic_axes", None),
                save_file=save_path,
            )
            ir_files.append(save_path)
        logger.info("Partition extraction completed: %d ONNX file(s).", len(ir_files))

    calib_filename = get_calib_filename(deploy_cfg)
    if calib_filename is not None:
        calib_path = osp.join(args.work_dir, calib_filename)
        create_calib_input_data(
            calib_path,
            deploy_cfg_path,
            model_cfg_path,
            checkpoint_path,
            dataset_cfg=args.calib_dataset_cfg,
            dataset_type="val",
            device=args.device,
        )
        logger.info("Calibration data generated: %s", calib_path)

    if not args.no_rewrite:
        _rewrite_onnx_files(ir_files, logger=logger, allow_custom_ops=args.allow_custom_ops)

    logger.info("Export complete. ONNX artifact(s): %s", ir_files)


if __name__ == "__main__":
    main()
