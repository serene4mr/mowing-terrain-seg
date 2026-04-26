"""
Shared ONNX custom-op → standard-op rewrite (mmdeploy/mmcv → ONNX standard).

Used by tools/deploy/deploy.py and tools/deploy/rewrite_custom_ops_onnx.py.
"""

from collections import defaultdict
from typing import Any, Dict, Tuple

import onnx
from onnx import helper

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


def rewrite_model_in_memory(model, logger=None):
    """Apply custom-op replacements to an ONNX model in memory.

    Returns:
        tuple: (model, replaced_count, kept_custom) where kept_custom maps
        (domain, op_type) -> count of nodes left as custom ops.
    """
    min_opset = max(_STANDARD_OPSET_REQUIRED.values())
    _ensure_opset(model, min_opset)

    new_nodes = []
    replaced_count = 0
    kept_custom: Dict[Tuple[str, str], int] = defaultdict(int)

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


def rewrite_custom_ops_to_file(
    path_in: str, path_out: str, logger=None
) -> Tuple[int, Dict[Tuple[str, str], int]]:
    """Load ONNX, rewrite custom ops, save to path_out. Returns (replaced_count, kept_custom)."""
    model = onnx.load(path_in)
    model, replaced_count, kept_custom = rewrite_model_in_memory(model, logger=logger)
    onnx.save(model, path_out)

    if logger:
        logger.info(
            f"Rewrite: saved {path_out}, replaced {replaced_count} custom op(s)."
        )
        if kept_custom:
            for (dom, op), count in sorted(kept_custom.items()):
                logger.info(f"  Kept {dom}::{op}: {count} node(s)")
    return replaced_count, dict(kept_custom)
