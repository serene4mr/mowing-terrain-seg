"""Tests for mmdeploy custom op -> standard ONNX rewrite."""
import tempfile

import onnx
from onnx import TensorProto, helper

import sys
import os

_DEPLOY = os.path.join(
    os.path.dirname(__file__), "..", "..", "tools", "deploy"
)
if _DEPLOY not in sys.path:
    sys.path.insert(0, _DEPLOY)
from _onnx_rewriter import rewrite_model_in_memory  # noqa: E402


def _minimal_grid_sampler_onnx() -> onnx.ModelProto:
    """One-input graph with mmdeploy::grid_sampler (rewritten to GridSample)."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
    grid = helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 2, 2, 2])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 2, 2])
    node = helper.make_node(
        "grid_sampler",
        ["X", "grid"],
        ["Y"],
        domain="mmdeploy",
        name="gs",
        interpolation_mode_i=0,
        padding_mode_i=0,
        align_corners_i=0,
    )
    graph = helper.make_graph([node], "g", [x, grid], [y])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 11)]
    )
    return model


def test_grid_sampler_rewritten_to_gridsample():
    model = _minimal_grid_sampler_onnx()
    new_model, replaced_count, kept = rewrite_model_in_memory(model, logger=None)
    assert replaced_count == 1
    assert not kept
    op_types = [n.op_type for n in new_model.graph.node]
    assert "GridSample" in op_types
    gs = [n for n in new_model.graph.node if n.op_type == "GridSample"][0]
    assert gs.domain == "" or gs.domain is None
    # Attributes from _rewrite_mmdeploy_grid_sampler
    names = {a.name for a in gs.attribute}
    assert "mode" in names and "padding_mode" in names and "align_corners" in names


def test_rewrite_custom_ops_to_file_roundtrip():
    from _onnx_rewriter import rewrite_custom_ops_to_file  # noqa: WPS433

    model = _minimal_grid_sampler_onnx()
    with tempfile.TemporaryDirectory() as d:
        p_in = os.path.join(d, "in.onnx")
        p_out = os.path.join(d, "out.onnx")
        onnx.save(model, p_in)
        rep, kept = rewrite_custom_ops_to_file(p_in, p_out, logger=None)
        assert rep == 1
        assert not kept
        m2 = onnx.load(p_out)
        assert any(n.op_type == "GridSample" for n in m2.graph.node)
