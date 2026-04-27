# tools/hf_release — Hugging Face release helpers

Internal modules used by **`tools/release.py`**. Do not call these directly; use the CLI instead.

## Modules

| Module | Responsibility |
|---|---|
| `validate.py` | Pre-flight: experiment dir, git, `.pth`↔ONNX drift, optional engine↔Hub ONNX hash |
| `staging.py` | Copies artifacts into the HF Hub repo layout under a temp dir |
| `engine.py` | Stages `tools/build_engine.py` output into `tensorrt/<profile>/` |
| `platform.py` | Jetson / L4T auto-detect for `platform.json` and profile names |
| `metrics.py` | Builds `metrics.json` from `summary.json` (+ optional eval pickle) |
| `card.py` | Renders the HF model card and merges the “Available TensorRT engines” table |

## Full workflow

```
tools/train.py          →  work_dirs/<exp>/summary.json + best_*.pth
tools/deploy/deploy.py  →  work_dirs/<exp>/deploy/onnx/end2end.onnx  (optional)
tools/release.py        →  huggingface.co/<org>/<repo>@<tag>
# On Jetson (after Hub has onnx/):
tools/build_engine.py   →  end2end.engine + platform.json
tools/release.py --engine-dir  →  same Hub: tensorrt/<profile>/
```

See [docs/mlops.md §4](../../docs/mlops.md) for the complete step-by-step guide including commands,
artifact layout, and the `.pth` ↔ ONNX matching rule. **Jetson / TensorRT:** [docs/deploy.md](../../docs/deploy.md) (TensorRT section).
