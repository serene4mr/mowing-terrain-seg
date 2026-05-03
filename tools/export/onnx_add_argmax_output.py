"""
Append an ArgMax output to an existing ONNX model.

This script takes an ONNX model that outputs raw logits (digits) and modifies it
to include a second output containing the predicted class indices (ArgMax).
This is useful for deployment scenarios where both the confidence scores
and the final mask are needed without running ArgMax on the client side.

Usage:
    python tools/export/onnx_add_argmax_output.py input.onnx output.onnx [options]
"""

import argparse
import onnx
from onnx import helper, TensorProto

def main():
    parser = argparse.ArgumentParser(description="Append an ArgMax output to an ONNX model")
    parser.add_argument("input_onnx", help="Path to the input ONNX model (logits output)")
    parser.add_argument("output_onnx", help="Path to save the modified ONNX model")
    parser.add_argument("--axis", type=int, default=1, help="Axis to compute ArgMax on (default: 1 for NCHW)")
    parser.add_argument("--drop-dims", action="store_true", help="Drop the channel dimension after ArgMax (default is to keep it)")
    args = parser.parse_args()

    print(f"Loading ONNX model from {args.input_onnx}...")
    model = onnx.load(args.input_onnx)

    # 1. Identify the existing logits output
    # Usually, the mmdeploy segmentation model has exactly one output (the logits)
    if len(model.graph.output) == 0:
        raise ValueError("The input ONNX model has no outputs!")
    
    logits_output = model.graph.output[0]
    logits_name = logits_output.name
    print(f"Found logits output: '{logits_name}'")

    # 2. Define the new ArgMax output
    argmax_name = logits_name + "_argmax"
    # We use INT64 for ArgMax output indices. 
    # Leaving shape as None lets ONNX runtime infer it dynamically.
    argmax_output_info = helper.make_tensor_value_info(
        argmax_name, 
        TensorProto.INT64, 
        None 
    )

    # 3. Create the ArgMax node
    keepdims_val = 0 if args.drop_dims else 1
    argmax_node = helper.make_node(
        "ArgMax",
        inputs=[logits_name],
        outputs=[argmax_name],
        axis=args.axis,
        keepdims=keepdims_val,
        name="PostProcess_ArgMax"
    )

    # 4. Append the node and the new output to the graph
    model.graph.node.append(argmax_node)
    model.graph.output.append(argmax_output_info)

    # 5. Save the modified model
    print(f"Saving dual-output ONNX model to {args.output_onnx}...")
    onnx.save(model, args.output_onnx)
    print("Done! The model now outputs both logits and argmax.")

if __name__ == "__main__":
    main()
