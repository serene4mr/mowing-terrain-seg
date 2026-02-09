"""
Replace mmdeploy::grid_sampler with standard ONNX GridSample (opset 13+).
Run: python tools/rewrite_grid_sampler_onnx.py \
  mmdeploy_model/mask2former-onnx/end2end.onnx \
  mmdeploy_model/mask2former-onnx/end2end_gridsample.onnx
"""
import argparse
import onnx
from onnx import helper

MODE_MAP = {0: "bilinear", 1: "nearest", 2: "bicubic"}
PADDING_MAP = {0: "zeros", 1: "border", 2: "reflection"}


def get_attr(node, name, default=0):
    for a in node.attribute:
        if a.name == name:
            return a.i
    return default


def rewrite(path_in: str, path_out: str):
    model = onnx.load(path_in)

    # Ensure opset 13+ for standard GridSample
    opset_domain = ""
    opset_version = 13
    for imp in model.opset_import:
        if imp.domain == opset_domain:
            if imp.version < opset_version:
                imp.version = opset_version
            break
    else:
        model.opset_import.append(helper.make_opsetid(opset_domain, opset_version))

    new_nodes = []
    replaced = 0
    for node in model.graph.node:
        if node.domain == "mmdeploy" and node.op_type == "grid_sampler":
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
            new_nodes.append(new_node)
            replaced += 1
        else:
            new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    onnx.save(model, path_out)
    print(f"Replaced {replaced} mmdeploy::grid_sampler nodes with GridSample. Saved to {path_out}")
    return replaced


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input_onnx", help="Path to end2end.onnx")
    p.add_argument("output_onnx", help="Path for rewritten ONNX")
    args = p.parse_args()
    rewrite(args.input_onnx, args.output_onnx)
