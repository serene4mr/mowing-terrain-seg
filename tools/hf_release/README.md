# tools/hf_release — Hugging Face release helpers

Internal modules used by **`tools/release.py`**. Do not call these directly; use the CLI instead.

## Modules

| Module | Responsibility |
|---|---|
| `validate.py` | Pre-flight checks: experiment dir layout, git cleanliness, `.pth` ↔ ONNX drift |
| `staging.py` | Copies artifacts into the HF Hub repo layout under a temp dir |
| `metrics.py` | Builds `metrics.json` from `summary.json` (+ optional eval pickle) |
| `card.py` | Renders the HF model card (`README.md`) as a hand-rolled f-string |

## Full workflow

```
tools/train.py          →  work_dirs/<exp>/summary.json + best_*.pth
tools/deploy/deploy.py  →  work_dirs/<exp>/deploy/onnx/end2end.onnx  (optional)
tools/release.py        →  huggingface.co/<org>/<repo>@<tag>
```

See [docs/mlops.md §4](../../docs/mlops.md) for the complete step-by-step guide including commands,
artifact layout, and the `.pth` ↔ ONNX matching rule.
