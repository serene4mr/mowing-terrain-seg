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
from collections import defaultdict

import onnx
from onnx import helper

# Custom domains we consider for replacement
CUSTOM_DOMAINS = ("mmdeploy", "mmcv")

# Standard op requirements: op_type -> minimum default-domain opset
STANDARD_OPSET_REQUIRED = {"GridSample": 13, "RoiAlign": 16}

MODE_MAP = {0: "bilinear", 1: "nearest", 2: "bicubic"}
PADDING_MAP = {0: "zeros", 1: "border", 2: "reflection"}


def get_attr(node, name, default=0):
    """Get integer attribute."""
    for a in node.attribute:
        if a.name == name:
            return a.i
    return default


def get_attr_f(node, name, default=0.0):
    """Get float attribute."""
    for a in node.attribute:
        if a.name == name:
            return a.f
    return default


def rewrite_mmdeploy_grid_sampler(node):
    """Replace mmdeploy::grid_sampler with standard GridSample."""
    interp = get_attr(node, "interpolation_mode_i", 0)
    padding = get_attr(node, "padding_mode_i", 0)
    align = get_attr(node, "align_corners_i", 0)

    mode_s = MODE_MAP.get(interp, "bilinear")
    padding_s = PADDING_MAP.get(padding, "zeros")
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


def get_attr_s(node, name, default=""):
    """Get string attribute."""
    for a in node.attribute:
        if a.name == name:
            if a.s:
                return a.s.decode("utf-8") if isinstance(a.s, bytes) else a.s
            return default
    return default


def rewrite_mmcv_roi_align(node):
    """Replace mmcv::MMCVRoIAlign (or mmcv::RoIAlign) with standard RoiAlign (opset 16+)."""
    out_h = get_attr(node, "output_height_i", get_attr(node, "aligned_height", 1))
    out_w = get_attr(node, "output_width_i", get_attr(node, "aligned_weight", 1))
    spatial_scale = get_attr_f(node, "spatial_scale_f", get_attr_f(node, "spatial_scale", 1.0))
    sampling_ratio = get_attr(node, "sampling_ratio_i", get_attr(node, "sampling_ratio", 0))
    mode = get_attr_s(node, "pool_mode") or get_attr_s(node, "mode_s") or "avg"
    if mode not in ("avg", "max"):
        mode = "avg"
    aligned = get_attr(node, "aligned", 0)
    coord_mode = "half_pixel" if aligned else "output_half_pixel"

    # Standard RoiAlign expects 3 inputs: X, rois, batch_indices
    if len(node.input) < 2:
        return []
    inputs = list(node.input)
    if len(inputs) == 2:
        # Exports with 2 inputs fold batch into rois; standard RoiAlign needs batch_indices.
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


# Registry: (domain, op_type) -> function(node) -> list of new nodes; return [] to keep original
REPLACEMENT_REGISTRY = {
    ("mmdeploy", "grid_sampler"): rewrite_mmdeploy_grid_sampler,
    ("mmcv", "MMCVRoIAlign"): rewrite_mmcv_roi_align,
    ("mmcv", "RoIAlign"): rewrite_mmcv_roi_align,
}


def ensure_opset(model, min_version):
    """Ensure default domain opset is at least min_version."""
    opset_domain = ""
    for imp in model.opset_import:
        if imp.domain == opset_domain:
            if imp.version < min_version:
                imp.version = min_version
            return
    model.opset_import.append(helper.make_opsetid(opset_domain, min_version))


def rewrite(path_in: str, path_out: str):
    model = onnx.load(path_in)

    # Determine minimum opset needed by any replacement we might apply
    min_opset = 1
    for req in STANDARD_OPSET_REQUIRED.values():
        min_opset = max(min_opset, req)
    ensure_opset(model, min_opset)

    new_nodes = []
    replaced_count = 0
    kept_custom = defaultdict(int)  # (domain, op_type) -> count

    for node in model.graph.node:
        domain = node.domain or ""
        op_type = node.op_type
        key = (domain, op_type)

        if domain in CUSTOM_DOMAINS:
            rewriter = REPLACEMENT_REGISTRY.get(key)
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
                    # On failure, keep original and report
                    new_nodes.append(node)
                    kept_custom[key] += 1
                    print(f"Warning: replacement failed for {domain}::{op_type} ({node.name}): {e}")
            else:
                new_nodes.append(node)
                kept_custom[key] += 1
        else:
            new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    onnx.save(model, path_out)

    # Report
    print(f"Saved to {path_out}")
    print(f"Replaced {replaced_count} custom op(s) with standard ONNX.")
    if kept_custom:
        print("Custom ops left as-is (no standard replacement):")
        for (dom, op), count in sorted(kept_custom.items()):
            print(f"  {dom}::{op}: {count} node(s)")

    return replaced_count


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Replace mmdeploy/mmcv custom ops with standard ONNX where possible."
    )
    p.add_argument("input_onnx", help="Path to input ONNX (e.g. end2end.onnx)")
    p.add_argument("output_onnx", help="Path for rewritten ONNX")
    args = p.parse_args()
    rewrite(args.input_onnx, args.output_onnx)
