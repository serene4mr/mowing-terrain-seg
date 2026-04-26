# MLOps Design — `mowing-terrain-seg` × Hugging Face Hub

Status: **Proposal / design doc** (not yet implemented)
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

Output (already produced by mmengine today):
```
work_dirs/dlv3p_r50_ycor_v1/
├── config.py                # saved by mmengine
├── *.pth                    # checkpoints
├── best_val_mIoU_iter_*.pth
├── scalars.json             # metrics over time
└── vis_data/
```

**Nothing is pushed yet.** Train as many candidates as you want.

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

This is the **one** new tool to add: `tools/release.py`.

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
1. Validates the experiment dir (`config.py`, `.pth`, `scalars.json` present).
2. Computes git SHA of **this** repo and refuses if dirty (`--allow-dirty` to override).
3. Builds a staging dir matching the HF layout (Section 3).
4. Generates `metrics.json` from `scalars.json` + optional eval pkl.
5. Generates `README.md` (model card) from a Jinja template + provenance.
6. `huggingface_hub.HfApi().upload_folder(...)` to the repo.
7. Creates a git tag on the HF repo (`create_tag`).
8. Prints the public URL.

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

### 5.1 New files

```
docs/mlops.md                            ← this document
tools/release.py                         ← promote work_dir → HF repo
tools/release/                           ← (optional) supporting modules
    ├── card_template.md.j2              ← Jinja template for model card
    ├── metrics.py                       ← scalars.json → metrics.json
    └── validate.py                      ← lint experiment before push
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

## 6. `tools/release.py` — interface sketch

```python
# tools/release.py
"""
Promote a local work_dirs/ experiment to a Hugging Face Hub repo as a tagged release.

Usage:
    python tools/release.py \
        --exp-dir work_dirs/<exp> \
        --pth     <ckpt_filename_inside_exp_dir> \
        --repo-id <org>/<repo> \
        --tag     v1.1 \
        [--deploy work_dirs/<exp>/deploy/onnx] \
        [--metrics work_dirs/<exp>/eval_results.json] \
        [--samples assets/image/sample.jpg]  \
        [--message "Release notes"] \
        [--dry-run] [--allow-dirty] [--private]
"""
```

Steps (pseudocode):

```python
def main(args):
    validate_experiment(args.exp_dir, args.pth)
    git_sha = check_git_clean(allow_dirty=args.allow_dirty)

    staging = build_staging_dir(args)        # copies into HF layout
    write_metrics_json(staging, args)
    render_model_card(staging, args, git_sha)

    if args.dry_run:
        print(f"[dry-run] would upload {staging} to {args.repo_id}@{args.tag}")
        return

    api = HfApi()
    api.create_repo(args.repo_id, private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=staging,
        commit_message=f"{args.tag}: {args.message}",
    )
    api.create_tag(args.repo_id, tag=args.tag, tag_message=args.message)
    print(f"https://huggingface.co/{args.repo_id}/tree/{args.tag}")
```

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
| 2 | Add `huggingface_hub` to `pyproject.toml` `[release]` extra | Dev |
| 3 | Implement `tools/release.py` (this doc) | Dev |
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
