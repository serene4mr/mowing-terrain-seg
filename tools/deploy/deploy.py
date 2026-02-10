"""
Deploy: export PyTorch model to ONNX (and other backends), then optionally
rewrite mmdeploy/mmcv custom ops to standard ONNX in place.

- Export: model config + checkpoint → ONNX (e.g. end2end.onnx) via mmdeploy.
- Rewrite (ONNX Runtime only): load ONNX, replace custom ops (e.g. grid_sampler
  → GridSample) in memory, save back to the same file. No temp file; one read,
  one write. Use --no-rewrite to skip and keep custom ops.

Usage:
  python tools/deploy/deploy.py <deploy_cfg> <model_cfg> <checkpoint> <image> [options]

Examples:
  # Export to ONNX and rewrite custom ops → work_dir/end2end.onnx
  python tools/deploy/deploy.py \\
    configs/deploy/custom/segmentation_onnxruntime_dynamic.py \\
    work_dirs/my_exp/config.py \\
    work_dirs/my_exp/best.pth \\
    assets/image/sample.jpg \\
    --work-dir mmdeploy_model/onnx \\
    --device cuda \\
    --dump-info

  # Export only, no rewrite (keep custom ops; need mmdeploy runtime)
  python tools/deploy/deploy.py ... --no-rewrite

  # Show visualization
  python tools/deploy/deploy.py ... --show
"""

import argparse
import logging
import os
import os.path as osp
from collections import defaultdict
from functools import partial

import mmengine
import onnx
import torch.multiprocessing as mp
from onnx import helper
from torch.multiprocessing import Process, set_start_method

from mmdeploy.apis import (create_calib_input_data, extract_model,
                           get_predefined_partition_cfg, torch2onnx,
                           torch2torchscript, visualize_model)
from mmdeploy.apis.core import PIPELINE_MANAGER
from mmdeploy.apis.utils import to_backend
from mmdeploy.backend.sdk.export_info import export2SDK
from mmdeploy.utils import (IR, Backend, get_backend, get_calib_filename,
                            get_ir_config, get_partition_config,
                            get_root_logger, load_config, target_wrapper)

# ----- Inlined rewrite logic (no import from rewrite_custom_ops_onnx) -----
_CUSTOM_DOMAINS = ("mmdeploy", "mmcv")
_STANDARD_OPSET_REQUIRED = {"GridSample": 13, "RoiAlign": 16}
_MODE_MAP = {0: "bilinear", 1: "nearest", 2: "bicubic"}
_PADDING_MAP = {0: "zeros", 1: "border", 2: "reflection"}


def _get_attr(node, name, default=0):
    for a in node.attribute:
        if a.name == name:
            return a.i
    return default


def _get_attr_f(node, name, default=0.0):
    for a in node.attribute:
        if a.name == name:
            return a.f
    return default


def _get_attr_s(node, name, default=""):
    for a in node.attribute:
        if a.name == name:
            if a.s:
                return a.s.decode("utf-8") if isinstance(a.s, bytes) else a.s
            return default
    return default


def _rewrite_mmdeploy_grid_sampler(node):
    interp = _get_attr(node, "interpolation_mode_i", 0)
    padding = _get_attr(node, "padding_mode_i", 0)
    align = _get_attr(node, "align_corners_i", 0)
    mode_s = _MODE_MAP.get(interp, "bilinear")
    padding_s = _PADDING_MAP.get(padding, "zeros")
    align_corners = 1 if align else 0
    new_node = helper.make_node(
        "GridSample",
        inputs=list(node.input),
        outputs=list(node.output),
        name=node.name + "_GridSample" if node.name else None,
        mode=mode_s,
        padding_mode=padding_s,
        align_corners=align_corners,
    )
    return [new_node]


def _rewrite_mmcv_roi_align(node):
    out_h = _get_attr(node, "output_height_i", _get_attr(node, "aligned_height", 1))
    out_w = _get_attr(node, "output_width_i", _get_attr(node, "aligned_weight", 1))
    spatial_scale = _get_attr_f(node, "spatial_scale_f", _get_attr_f(node, "spatial_scale", 1.0))
    sampling_ratio = _get_attr(node, "sampling_ratio_i", _get_attr(node, "sampling_ratio", 0))
    mode = _get_attr_s(node, "pool_mode") or _get_attr_s(node, "mode_s") or "avg"
    if mode not in ("avg", "max"):
        mode = "avg"
    aligned = _get_attr(node, "aligned", 0)
    coord_mode = "half_pixel" if aligned else "output_half_pixel"
    if len(node.input) < 2:
        return []
    inputs = list(node.input)
    if len(inputs) == 2:
        return []
    new_node = helper.make_node(
        "RoiAlign",
        inputs=inputs,
        outputs=list(node.output),
        name=node.name + "_RoiAlign" if node.name else None,
        mode=mode,
        output_height=out_h,
        output_width=out_w,
        sampling_ratio=sampling_ratio,
        spatial_scale=spatial_scale,
        coordinate_transformation_mode=coord_mode,
    )
    return [new_node]


_REPLACEMENT_REGISTRY = {
    ("mmdeploy", "grid_sampler"): _rewrite_mmdeploy_grid_sampler,
    ("mmcv", "MMCVRoIAlign"): _rewrite_mmcv_roi_align,
    ("mmcv", "RoIAlign"): _rewrite_mmcv_roi_align,
}


def _ensure_opset(model, min_version):
    opset_domain = ""
    for imp in model.opset_import:
        if imp.domain == opset_domain:
            if imp.version < min_version:
                imp.version = min_version
            return
    model.opset_import.append(helper.make_opsetid(opset_domain, min_version))


def _rewrite_model_in_memory(model, logger=None):
    """Apply custom-op replacements to an ONNX model in memory. Returns (model, replaced_count, kept_custom)."""
    min_opset = max(_STANDARD_OPSET_REQUIRED.values())
    _ensure_opset(model, min_opset)

    new_nodes = []
    replaced_count = 0
    kept_custom = defaultdict(int)

    for node in model.graph.node:
        domain = node.domain or ""
        op_type = node.op_type
        key = (domain, op_type)

        if domain in _CUSTOM_DOMAINS:
            rewriter = _REPLACEMENT_REGISTRY.get(key)
            if rewriter is not None:
                try:
                    replacement_nodes = rewriter(node)
                    if replacement_nodes:
                        new_nodes.extend(replacement_nodes)
                        replaced_count += 1
                    else:
                        new_nodes.append(node)
                        kept_custom[key] += 1
                except Exception as e:
                    new_nodes.append(node)
                    kept_custom[key] += 1
                    if logger:
                        logger.warning(
                            f"Replacement failed for {domain}::{op_type} ({node.name}): {e}"
                        )
            else:
                new_nodes.append(node)
                kept_custom[key] += 1
        else:
            new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model, replaced_count, kept_custom


def rewrite_custom_ops(path_in: str, path_out: str, logger=None):
    """Replace mmdeploy/mmcv custom ops with standard ONNX. Inlined (no import)."""
    model = onnx.load(path_in)
    model, replaced_count, kept_custom = _rewrite_model_in_memory(model, logger=logger)
    onnx.save(model, path_out)

    if logger:
        logger.info(f"Rewrite: saved {path_out}, replaced {replaced_count} custom op(s).")
        if kept_custom:
            for (dom, op), count in sorted(kept_custom.items()):
                logger.info(f"  Kept {dom}::{op}: {count} node(s)")
    return replaced_count


# ----- Deploy (same as deploy.py) -----
def parse_args():
    parser = argparse.ArgumentParser(
        description='Export model to backends, then rewrite custom ops to standard ONNX.'
    )
    parser.add_argument('deploy_cfg', help='deploy config path')
    parser.add_argument('model_cfg', help='model config path')
    parser.add_argument('checkpoint', help='model checkpoint path')
    parser.add_argument('img', help='image used to convert model model')
    parser.add_argument(
        '--test-img',
        default=None,
        type=str,
        nargs='+',
        help='image used to test model')
    parser.add_argument(
        '--work-dir',
        default=os.getcwd(),
        help='the dir to save logs and models')
    parser.add_argument(
        '--calib-dataset-cfg',
        help='dataset config path used to calibrate in int8 mode. If not \
            specified, it will use "val" dataset in model config instead.',
        default=None)
    parser.add_argument(
        '--device', help='device used for conversion', default='cpu')
    parser.add_argument(
        '--log-level',
        help='set log level',
        default='INFO',
        choices=list(logging._nameToLevel.keys()))
    parser.add_argument(
        '--show', action='store_true', help='Show detection outputs')
    parser.add_argument(
        '--dump-info', action='store_true', help='Output information for SDK')
    parser.add_argument(
        '--quant-image-dir',
        default=None,
        help='Image directory for quantize model.')
    parser.add_argument(
        '--quant', action='store_true', help='Quantize model to low bit.')
    parser.add_argument(
        '--uri',
        default='192.168.1.1:60000',
        help='Remote ipv4:port or ipv6:port for inference on edge device.')
    parser.add_argument(
        '--no-rewrite',
        action='store_true',
        help='Skip rewriting custom ops to standard ONNX (only run deploy).')
    args = parser.parse_args()
    return args


def create_process(name, target, args, kwargs, ret_value=None):
    logger = get_root_logger()
    logger.info(f'{name} start.')
    log_level = logger.level

    wrap_func = partial(target_wrapper, target, log_level, ret_value)

    process = Process(target=wrap_func, args=args, kwargs=kwargs)
    process.start()
    process.join()

    if ret_value is not None:
        if ret_value.value != 0:
            logger.error(f'{name} failed.')
            exit(1)
        else:
            logger.info(f'{name} success.')


def torch2ir(ir_type: IR):
    if ir_type == IR.ONNX:
        return torch2onnx
    elif ir_type == IR.TORCHSCRIPT:
        return torch2torchscript
    else:
        raise KeyError(f'Unexpected IR type {ir_type}')


def main():
    args = parse_args()
    set_start_method('spawn', force=True)
    logger = get_root_logger()
    log_level = logging.getLevelName(args.log_level)
    logger.setLevel(log_level)

    pipeline_funcs = [
        torch2onnx, torch2torchscript, extract_model, create_calib_input_data
    ]
    PIPELINE_MANAGER.enable_multiprocess(True, pipeline_funcs)
    PIPELINE_MANAGER.set_log_level(log_level, pipeline_funcs)

    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg
    checkpoint_path = args.checkpoint
    quant = args.quant
    quant_image_dir = args.quant_image_dir

    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)

    mmengine.mkdir_or_exist(osp.abspath(args.work_dir))

    if args.dump_info:
        export2SDK(
            deploy_cfg,
            model_cfg,
            args.work_dir,
            pth=checkpoint_path,
            device=args.device)

    ret_value = mp.Value('d', 0, lock=False)

    ir_config = get_ir_config(deploy_cfg)
    ir_save_file = ir_config['save_file']
    ir_type = IR.get(ir_config['type'])
    torch2ir(ir_type)(
        args.img,
        args.work_dir,
        ir_save_file,
        deploy_cfg_path,
        model_cfg_path,
        checkpoint_path,
        device=args.device)

    ir_files = [osp.join(args.work_dir, ir_save_file)]

    partition_cfgs = get_partition_config(deploy_cfg)

    if partition_cfgs is not None:

        if 'partition_cfg' in partition_cfgs:
            partition_cfgs = partition_cfgs.get('partition_cfg', None)
        else:
            assert 'type' in partition_cfgs
            partition_cfgs = get_predefined_partition_cfg(
                deploy_cfg, partition_cfgs['type'])

        origin_ir_file = ir_files[0]
        ir_files = []
        for partition_cfg in partition_cfgs:
            save_file = partition_cfg['save_file']
            save_path = osp.join(args.work_dir, save_file)
            start = partition_cfg['start']
            end = partition_cfg['end']
            dynamic_axes = partition_cfg.get('dynamic_axes', None)

            extract_model(
                origin_ir_file,
                start,
                end,
                dynamic_axes=dynamic_axes,
                save_file=save_path)

            ir_files.append(save_path)

    calib_filename = get_calib_filename(deploy_cfg)
    if calib_filename is not None:
        calib_path = osp.join(args.work_dir, calib_filename)
        create_calib_input_data(
            calib_path,
            deploy_cfg_path,
            model_cfg_path,
            checkpoint_path,
            dataset_cfg=args.calib_dataset_cfg,
            dataset_type='val',
            device=args.device)

    backend_files = ir_files
    backend = get_backend(deploy_cfg)

    if backend == Backend.RKNN:
        import tempfile

        from mmdeploy.utils import (get_common_config, get_normalization,
                                    get_quantization_config,
                                    get_rknn_quantization)
        quantization_cfg = get_quantization_config(deploy_cfg)
        common_params = get_common_config(deploy_cfg)
        if get_rknn_quantization(deploy_cfg) is True:
            transform = get_normalization(model_cfg)
            common_params.update(
                dict(
                    mean_values=[transform['mean']],
                    std_values=[transform['std']]))

        dataset_file = tempfile.NamedTemporaryFile(suffix='.txt').name
        with open(dataset_file, 'w') as f:
            f.writelines([osp.abspath(args.img)])
        if quantization_cfg.get('dataset', None) is None:
            quantization_cfg['dataset'] = dataset_file
    if backend == Backend.ASCEND:
        if args.dump_info:
            from mmdeploy.backend.ascend import update_sdk_pipeline
            update_sdk_pipeline(args.work_dir)

    if backend == Backend.VACC:
        from onnx2vacc_quant_dataset import get_quant

        from mmdeploy.utils import get_model_inputs

        deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)
        model_inputs = get_model_inputs(deploy_cfg)

        for onnx_path, model_input in zip(ir_files, model_inputs):

            quant_mode = model_input.get('qconfig', {}).get('dtype', 'fp16')
            assert quant_mode in ['int8',
                                  'fp16'], quant_mode + ' not support now'
            shape_dict = model_input.get('shape', {})

            if quant_mode == 'int8':
                create_process(
                    'vacc quant dataset',
                    target=get_quant,
                    args=(deploy_cfg, model_cfg, shape_dict, checkpoint_path,
                          args.work_dir, args.device),
                    kwargs=dict(),
                    ret_value=ret_value)

    PIPELINE_MANAGER.set_log_level(log_level, [to_backend])
    if backend == Backend.TENSORRT:
        PIPELINE_MANAGER.enable_multiprocess(True, [to_backend])
    backend_files = to_backend(
        backend,
        ir_files,
        work_dir=args.work_dir,
        deploy_cfg=deploy_cfg,
        log_level=log_level,
        device=args.device,
        uri=args.uri)

    # Rewrite custom ops in memory, then write once to the same ONNX file (no temp file)
    if backend == Backend.ONNXRUNTIME and not args.no_rewrite:
        for bf in backend_files:
            if bf.endswith('.onnx') and osp.isfile(bf):
                model = onnx.load(bf)
                model, replaced_count, kept_custom = _rewrite_model_in_memory(model, logger=logger)
                onnx.save(model, bf)
                logger.info(
                    f"Rewritten (standard ops) in place: {bf}, replaced {replaced_count} custom op(s)."
                )
                if kept_custom:
                    for (dom, op), count in sorted(kept_custom.items()):
                        logger.info(f"  Kept {dom}::{op}: {count} node(s)")

    if backend == Backend.NCNN and quant:
        from onnx2ncnn_quant_table import get_table

        from mmdeploy.apis.ncnn import get_quant_model_file, ncnn2int8
        model_param_paths = backend_files[::2]
        model_bin_paths = backend_files[1::2]
        backend_files = []
        for onnx_path, model_param_path, model_bin_path in zip(
                ir_files, model_param_paths, model_bin_paths):

            deploy_cfg, model_cfg = load_config(deploy_cfg_path,
                                                model_cfg_path)
            quant_onnx, quant_table, quant_param, quant_bin = get_quant_model_file(  # noqa: E501
                onnx_path, args.work_dir)

            create_process(
                'ncnn quant table',
                target=get_table,
                args=(onnx_path, deploy_cfg, model_cfg, quant_onnx,
                      quant_table, quant_image_dir, args.device),
                kwargs=dict(),
                ret_value=ret_value)

            create_process(
                'ncnn_int8',
                target=ncnn2int8,
                args=(model_param_path, model_bin_path, quant_table,
                      quant_param, quant_bin),
                kwargs=dict(),
                ret_value=ret_value)
            backend_files += [quant_param, quant_bin]

    if args.test_img is None:
        args.test_img = args.img

    extra = dict(
        backend=backend,
        output_file=osp.join(args.work_dir, f'output_{backend.value}.jpg'),
        show_result=args.show)
    if backend == Backend.SNPE:
        extra['uri'] = args.uri

    if args.show:
        create_process(
            f'visualize {backend.value} model',
            target=visualize_model,
            args=(model_cfg_path, deploy_cfg_path, backend_files,
                  args.test_img, args.device),
            kwargs=extra,
            ret_value=ret_value)

        create_process(
            'visualize pytorch model',
            target=visualize_model,
            args=(model_cfg_path, deploy_cfg_path, [checkpoint_path],
                  args.test_img, args.device),
            kwargs=dict(
                backend=Backend.PYTORCH,
                output_file=osp.join(args.work_dir, 'output_pytorch.jpg'),
                show_result=args.show),
            ret_value=ret_value)
    logger.info('All process success.')


if __name__ == '__main__':
    main()
