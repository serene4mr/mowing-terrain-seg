# Deploy Tools

This folder contains model export and ONNX custom-op rewrite utilities.

## Why rewriting is needed

When exporting through mmdeploy, the resulting ONNX model may contain
**custom ops** from the `mmdeploy` or `mmcv` domains (e.g. `mmdeploy::grid_sampler`).
These ops are not part of the standard ONNX spec and require mmdeploy's
custom-op runtime library (`libmmdeploy_onnxruntime_ops.so`) to run.

If you want to run the model with **standard ONNX Runtime** (without mmdeploy
runtime installed, e.g. on a different machine or in production), the custom
ops must be replaced with their standard ONNX equivalents.

The rewrite step does this in-place — no model retraining or re-export needed.

## Files

- `deploy.py`: Main export pipeline (PyTorch -> backend artifacts). For ONNX Runtime, it can rewrite known custom ops in-place after export.
- `rewrite_custom_ops_onnx.py`: Standalone CLI to rewrite an existing ONNX file.
- `_onnx_rewriter.py`: Shared rewrite core (op mappings, graph rewrite, registry API).

## Typical usage

### 1) Export + rewrite in one step

```bash
python tools/deploy/deploy.py \
  <deploy_cfg> <model_cfg> <checkpoint> <img> \
  --work-dir <out_dir>
```

### 2) Rewrite an existing ONNX only

```bash
python tools/deploy/rewrite_custom_ops_onnx.py input.onnx output.onnx
```

If custom ops remain after rewrite, the script exits with code `2`.
Use `--allow-custom-ops` to keep exit code `0`.

## Extending rewrite support

The rewriter is intentionally targeted (not generic for all custom ops).
To add support for a new custom op, register a handler in `_onnx_rewriter.py`:

```python
from tools.deploy._onnx_rewriter import register_rewriter

def my_rewriter(node):
    # return list of replacement ONNX nodes
    return [new_node]

register_rewriter("my_domain", "MyCustomOp", my_rewriter)
```

The core rewrite flow will pick up registered handlers automatically.
