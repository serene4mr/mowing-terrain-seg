# MLOps Design — `mowing-terrain-seg` × Hugging Face Hub

Status: **MLOps design** — `tools/release.py` and `work_dirs/.../summary.json` are implemented; HF promotion is **manual** (opt-in `pip install -e ".[release]"`)
Scope: training → evaluation → deploy export → release → consumption
Artifact store: **Hugging Face Hub** (one repo per model variant)
Code host: **GitHub** (this repo)

---

## 1. Goals & non-goals

### Goals
- Decouple **code** (versioned in GitHub) from **artifacts** (versioned on HF Hub).
- Make every released model **reproducible**: given a HF revision, you can find the exact config, commit, and dataset split that produced it.
- Keep the **inner loop fast**: experiments stay local; only promoted runs go remote.
- Support both **PyTorch (.pth)** and **deployed (ONNX)** artifacts in the same release.
- Zero infra to maintain — no S3 buckets, no MLflow server, no DVC remote setup.

### Non-goals
- Replacing live training metric tracking (MLflow/W&B can be added orthogonally later).
- Multi-tenant access control (HF private repos are enough).
- Continuous training / auto-retraining on data drift.
- Auto-deploy to a serving endpoint (out of scope for v1).

---

## 2. Architecture overview

```
┌────────────────────┐                ┌──────────────────────┐
│       GitHub       │                │   Hugging Face Hub   │
│ mowing-terrain-seg │                │  (artifact store)    │
│                    │                │                      │
│  - configs/        │   release      │  hf.co/<org>/        │
│  - src/            │ ─────────────► │     mts-<model>      │
│  - tools/          │  push_artifact │       └─ v1.0 (tag)  │
│  - tests/          │                │       └─ v1.1 (tag)  │
│  - docs/           │ ◄───────────── │                      │
│                    │  pull_artifact │  hf.co/<org>/        │
└─────────┬──────────┘                │     mts-<model2>     │
          │                           └──────────────────────┘
          │ developer
          │ runs locally / on GPU box
          ▼
   ┌─────────────────────┐
   │  work_dirs/<exp>/   │   ← local-only, dirty, many runs
   │   ├ best.pth        │     not tracked in git
   │   ├ config.py       │     not pushed to HF
   │   ├ scalars.json    │
   │   └ vis_data/       │
   └─────────────────────┘
```

### Three storage layers

| Layer | Where | Purpose | Lifetime |
|---|---|---|---|
| **Code** | GitHub | configs, scripts, tests | forever, versioned |
| **Workspace** | local `work_dirs/` | every experiment | days–weeks, disposable |
| **Release** | HF Hub | promoted models only | forever, versioned by tag |

---

## 3. HF Hub repository layout

### One HF repo per **model variant**

```
hf.co/<org>/mts-deeplabv3plus-r50-ycor3cls
hf.co/<org>/mts-mask2former-r50-ycor3cls
hf.co/<org>/mts-deeplabv3-r101-ycor3cls
```

Naming convention: `mts-<arch>-<backbone>-<dataset><classes>`.

### Files inside each HF repo

```
mts-<model>/
├── README.md                # HF model card (metadata + usage)
├── config.py                # exact training config (copied from work_dirs)
├── pytorch/
│   └── best.pth             # promoted checkpoint (LFS)
├── onnx/
│   ├── end2end.onnx         # rewritten, ORT-runnable (LFS)
│   ├── pipeline.json        # mmdeploy SDK metadata
│   ├── deploy.json
│   └── detail.json          # provenance: which .pth + config produced this
├── samples/
│   ├── input.jpg            # one demo image
│   └── output.png           # expected output
└── metrics.json             # mIoU, per-class IoU, val set, eval date
```

### Versioning via git tags

HF repos are git repos. Use **tags** as the version axis.

```
v1.0    initial release, deeplabv3plus, mIoU 0.78 on YCOR-val
v1.1    retrained with class-frequency weights, mIoU 0.81
v2.0    ONNX exported, opset 17, ORT-compatible
v2.1    bug fix in pipeline.json resize cfg
```

Consumers pin a revision:

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download(
    repo_id="<org>/mts-deeplabv3plus-r50-ycor3cls",
    filename="pytorch/best.pth",
    revision="v1.1",
)
```

### Model card schema (required fields)

```yaml
---
license: apache-2.0
library_name: mmsegmentation
tags:
  - semantic-segmentation
  - off-road
  - autonomous-mowing
datasets:
  - ycor
metrics:
  - mIoU
model-index:
  - name: mts-deeplabv3plus-r50-ycor3cls
    results:
      - task:
          type: image-segmentation
        dataset:
          name: YCOR-val
          type: ycor
        metrics:
          - type: mIoU
            value: 0.812
---

# mowing-terrain-seg / DeepLabV3+ R50 / YCOR 3-class

3-class off-road segmentation: `cuttable / traversable / non-traversable`.

## Provenance
- Source repo: https://github.com/<org>/mowing-terrain-seg @ commit `abc1234`
- Config: `configs/train/ycor-lm-3cls-exps/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-40k_ycor-1024x544.py`
- Trained: 40000 iter, 4×V100, AMP on
- Date: 2026-04-26

## Usage
... see "Consumption" section below ...
```

---

## 4. End-to-end workflow

### Stage A: Train (local / GPU box)

```bash
python tools/train.py \
    configs/train/ycor-lm-3cls-exps/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-40k_ycor-1024x544.py \
    --work-dir work_dirs/dlv3p_r50_ycor_v1
```

Output (mmengine run layout under the experiment work dir):
```
work_dirs/dlv3p_r50_ycor_v1/
├── <YYYYMMDD_HHMMSS>/       # one folder per start/resume; latest is what summarizer reads
│   ├── <YYYYMMDD_HHMMSS>.log
│   └── vis_data/
│       ├── scalars.json     # jsonl: train + val (canonical)
│       ├── config.py        # resolved config snapshot
│       └── 2026....json     # duplicate of scalars (mmengine; ignore in tooling)
├── *.py                      # e.g. copy of the training config at work_dir root
├── best_val_mIoU_iter_*.pth
├── iter_*.pth, last_checkpoint
└── summary.json              # **written at end of `tools/train.py` (or via CLI below)**
```

After a successful `runner.train()` call, the repo automatically runs `tools.summarize_experiment.summarize(work_dir)`.

`summary.json` (schema v1) consolidates: best/last val metrics, downsampled `history`, dataset metadata from `vis_data/config.py`, `config_sha256` of the work_dir root `*.py` config, git `sha` (when in a git checkout), and environment versions. Per-class IoU/Acc (when available) is parsed from the **train** `<run>.log` (not from `scalars.json`).

**Retroactive / manual:**

```bash
python tools/summarize_experiment.py work_dirs/dlv3p_r50_ycor_v1
```

Use `--no-summarize` on `tools/train.py` to skip (e.g. fast smoke tests). Summary failures are logged as a warning and do not fail the training run.

**Nothing is pushed to Hugging Face yet.** Train as many candidates as you want; promotion is a separate step.

### Stage B: Evaluate

```bash
python tools/test.py \
    work_dirs/dlv3p_r50_ycor_v1/config.py \
    work_dirs/dlv3p_r50_ycor_v1/best_val_mIoU_iter_38000.pth \
    --out work_dirs/dlv3p_r50_ycor_v1/eval_results.pkl
```

Eval results stay in `work_dirs/`. Compare candidates manually or via a helper script.

### Stage C: Deploy export (optional, before release)

```bash
python tools/deploy/deploy.py \
    configs/deploy/custom/segmentation_onnxruntime_dynamic.py \
    work_dirs/dlv3p_r50_ycor_v1/config.py \
    work_dirs/dlv3p_r50_ycor_v1/best_val_mIoU_iter_38000.pth \
    assets/image/sample.jpg \
    --work-dir work_dirs/dlv3p_r50_ycor_v1/deploy/onnx \
    --dump-info
```

This writes:
```
work_dirs/dlv3p_r50_ycor_v1/deploy/onnx/
├── end2end.onnx
├── pipeline.json
├── deploy.json
└── detail.json     # already references the source .pth + config
```

`detail.json` is your **automatic provenance link** between deploy artifacts and the training run.

### Stage D: Release (manual, deliberate)

**Implemented:** [tools/release.py](tools/release.py) (opt-in: `pip install -e ".[release]"`).

```bash
python tools/release.py \
    --exp-dir work_dirs/dlv3p_r50_ycor_v1 \
    --pth      best_val_mIoU_iter_38000.pth \
    --deploy   work_dirs/dlv3p_r50_ycor_v1/deploy/onnx \
    --repo-id  <org>/mts-deeplabv3plus-r50-ycor3cls \
    --tag      v1.1 \
    --message  "Retrained with class-frequency weights"
```

What it does:
1. Validates the experiment dir: top-level training `*.py` config, `summary.json`, and the chosen `.pth`.
2. Records git SHA of **this** code repo; refuses if unknown/dirty (use `--allow-dirty` to override).
3. Checks **drift** between the `.pth` and `onnx/` if `--deploy` is set (Section 6.1).
4. Builds a staging dir matching the HF layout (Section 3): `config.py`, `summary.json`, `pytorch/best.pth`, `onnx/`, `README.md`, `metrics.json` (+ optional `samples/`).
5. `HfApi().create_repo` (private by default) + `upload_folder` + `create_tag` on the Hub.
6. Writes `work_dirs/<exp>/release.json` with provenance, prints `https://huggingface.co/.../tree/<tag>`.

### Stage E: Consume (downstream / production)

```python
from huggingface_hub import hf_hub_download
from mowing_terrain_seg.inference.predictor import SegPredictor, Backend
import mowing_terrain_seg

mowing_terrain_seg.register_all()

REPO = "<org>/mts-deeplabv3plus-r50-ycor3cls"
REV  = "v1.1"

cfg  = hf_hub_download(REPO, "config.py",        revision=REV)
ckpt = hf_hub_download(REPO, "pytorch/best.pth", revision=REV)

pred = SegPredictor(cfg_uri=cfg, model_uri=ckpt, backend=Backend.TORCH)
mask = pred.predict("input.jpg")
```

For ONNX:

```python
from huggingface_hub import snapshot_download
local = snapshot_download(REPO, revision=REV, allow_patterns=["onnx/*"])
pred = SegPredictor(
    cfg_uri=f"{local}/onnx/pipeline.json",
    model_uri=f"{local}/onnx/end2end.onnx",
    backend=Backend.ONNX,
)
```

`huggingface_hub` caches under `~/.cache/huggingface/hub/` — no extra logic needed.

---

## 5. Repo changes required

### 5.1 New files (implemented)

```
docs/mlops.md                            ← this document
tools/release.py                         ← promote work_dir → HF repo (CLI)
tools/hf_release/                        ← release helpers
    ├── validate.py                      ← experiment + drift checks
    ├── staging.py                     ← build HF model-repo layout
    ├── metrics.py                     ← `summary.json` (+ optional pkl) → `metrics.json`
    └── card.py                        ← `README.md` (model card) as hand-rolled template
```

### 5.2 Modified files

| File | Change |
|---|---|
| `pyproject.toml` | add `huggingface_hub>=0.20` to optional `release` extra |
| `requirements.txt` | (no change — release tooling is opt-in) |
| `.gitignore` | already excludes `work_dirs/`, `mmdeploy_model/`, fine |
| `README.md` | add "Releasing models" section linking here |
| `.github/workflows/ci.yml` | (optional) add `release-validate` job that runs `tools/release.py --dry-run` on a fixture |

### 5.3 Optional cleanups

- Move `mmdeploy_model/` outputs **into** their experiment dir under `work_dirs/<exp>/deploy/` so `detail.json` provenance stays local. Top-level `mmdeploy_model/` becomes deprecated.

---

## 6. `tools/release.py` — interface (implemented)

Install the optional extra: `pip install -e ".[release]"` (adds `huggingface_hub`).

**Minimum (PyTorch + config + `summary.json` + metrics + model card):**

```bash
python tools/release.py \
    --exp-dir work_dirs/<exp> \
    --pth     <ckpt_filename_inside_exp_dir> \
    --repo-id <org>/mts-... \
    --tag     v1.0.0 \
    --message "Release notes" \
    [--allow-dirty]
```

Hugging Face auth uses the normal `huggingface_hub` flow (`HF_TOKEN` or `huggingface-cli login`). New repos are **private** by default; pass `--public` for a public model repo.

**With ONNX** (after a local `tools/deploy/deploy.py` run): pass `--deploy` to the directory that contains `end2end.onnx` (e.g. `work_dirs/<exp>/deploy/onnx`). ONNX conversion is **always a separate step** — run `tools/deploy/deploy.py` first, then pass the output dir to `--deploy`.

**Other flags:**

- `--metrics-pkl` — optional eval pickle; merged into `metrics.json` as `eval_extra` (shallow, JSON-safe)
- `--samples-in` + `--samples-out` — copy demo images to `samples/input.jpg` and `samples/output.png`
- `--dry-run` — build the release tree in a temp dir and list files; no Hub call (no `huggingface_hub` required)

**Exit codes:** `0` success, `2` validation (missing files, git policy), `3` Hugging Face API error.

**Local provenance after a real upload:** `work_dirs/<exp>/release.json` (repo, tag, pth, git sha, has_onnx, time).

### 6.1 Drift safety (`.pth` vs ONNX)

If `--deploy` is set, the tool requires `end2end.onnx` and **fails** if the checkpoint is **newer** than the ONNX (mtime), with a one-line hint to re-run `tools/deploy/deploy.py`. This keeps published `onnx/` aligned with `pytorch/best.pth` on the same tag.

`detail.json` is checked best-effort: if the checkpoint name does not appear, a **warning** is logged (mmdeploy’s schema can vary by version). Re-run deploy when in doubt.

---

## 7. Provenance & traceability

For every released artifact, you can answer **"how was this made?"** in one click:

```
HF repo @ tag v1.1
   └─ README.md      → links to GitHub commit abc1234
   └─ config.py      → exact training config
   └─ metrics.json   → val set + scores at release time
   └─ onnx/detail.json → references the source .pth + config
                                      ↑
                                      │ this chain works because
                                      │ deploy.py already writes detail.json
```

Reverse direction (given a deployed ONNX in the field, find the source):

```
end2end.onnx  →  detail.json  →  pth filename + config path
                                     ↓
                              (you keep the work_dir locally
                               OR you re-download from the same
                               HF tag's pytorch/best.pth)
```

---

## 8. Security & access

- **Public repos** for open-sourced models. Anyone can `pip install + hf_hub_download`.
- **Private repos** for proprietary weights. Set `HF_TOKEN` env var; works transparently.
- **Write tokens** only on developer machines; CI does not need write tokens unless you choose to auto-promote (not recommended).

---

## 9. Migration plan

| Step | Action | Owner |
|---|---|---|
| 1 | Create empty HF repos for each existing model variant | Maintainer |
| 2 | `huggingface_hub` on optional `[release]` extra in `pyproject.toml` (done) | Dev |
| 3 | `tools/release.py` + `tools/hf_release/` (done) | Dev |
| 4 | Dry-run release on one current experiment | Dev |
| 5 | Promote first real release (`v1.0`) | Maintainer |
| 6 | Update `README.md` with "Loading from HF" snippet | Dev |
| 7 | Optional: deprecate top-level `mmdeploy_model/` in favor of `work_dirs/<exp>/deploy/` | Dev |

---

## 10. Decisions log

| Decision | Choice | Rationale |
|---|---|---|
| Artifact store | HF Hub | free, versioned, well-known, zero infra |
| Repo granularity | one repo per model variant | clean cards, separate version histories |
| Versioning | git tags on HF repo | native to git LFS, no custom scheme |
| Promotion trigger | manual `tools/release.py` | training is too expensive for auto-CI |
| Metrics tracking | local `scalars.json` + per-release `metrics.json` | MLflow optional, not required |
| Deploy artifacts | included in same HF repo under `onnx/` | one revision = full deliverable |
| `detail.json` | kept as-is (mmdeploy default) | already provides provenance link |
| Code dirty check | block release unless `--allow-dirty` | guarantees git SHA in card is meaningful |

---

## 11. Open questions

- Do we want a **single combined repo** for all variants under one org account vs. spreading them? *Recommendation: separate, but easy to revisit.*
- Do we host a small **`mts-datasets`** repo for the YCOR split definitions (json with file lists), so eval is reproducible? *Recommendation: yes, low cost.*
- Do we want an `mts-demo` Space (Gradio) auto-pulling the latest release? *Out of scope for v1, but trivial later.*
- Should `tools/release.py` also push a `latest` floating tag for convenience? *Yes, but only after `v1.0` is stable.*

---

## 12. TL;DR for contributors

1. Train as usual → `work_dirs/<exp>/`.
2. Happy with results? Run:
   ```bash
   python tools/release.py --exp-dir work_dirs/<exp> --pth best.pth \
       --repo-id <org>/mts-<model> --tag vX.Y --message "..."
   ```
3. Done. The model is now downloadable, versioned, documented, and traceable.
