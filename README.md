# Mowing Terrain Segmentation

A semantic segmentation benchmark for off-road navigation and autonomous lawn mowing applications, built on MMSegmentation framework.

## Overview

This project provides tools and models for semantic segmentation of off-road terrain, with a focus on autonomous lawn mowing. It supports 3-class segmentation (Cuttable/Traversable/Non-Traversable) for safe navigation and obstacle avoidance.

## Features

- **Models**: DeepLabV3 and DeepLabV3+ with ResNet-50 backbone
- **Datasets**: YCOR (Yamaha) and Rellis-3D support
- **Training**: Configurable training pipelines with weighted loss functions
- **Inference**: Image, video, and batch processing with visualization
- **Analysis**: Dataset analysis and visualization tools

## Installation

Install PyTorch, MMCV, and OpenMMLab stack (see `requirements.txt` for exact pins, including the MMCV prebuilt wheel URL), then install this repo in editable mode so `mowing_terrain_seg` and CLI scripts resolve imports without `sys.path` hacks:

```bash
pip install -r requirements.txt
pip install -e .
```

Optional dev tools: `pip install -e ".[dev]"` (pytest, ruff, onnx for tests).

**Releasing models to Hugging Face Hub:** opt-in with `pip install -e ".[release]"` (adds `huggingface_hub`), then e.g.:

```bash
python tools/release.py --exp-dir work_dirs/<your_exp> --pth best_val_mIoU_iter_5000.pth \
  --repo-id <org>/mts-... --tag v1.0.0 --message "Notes" --dry-run --allow-dirty
```

Use `--deploy work_dirs/.../deploy/onnx` to include a prior ONNX export; see **§6** in [docs/mlops.md](docs/mlops.md) for the full flag list, drift check, and auth.

## Quick Start

### Training

Train a segmentor.

```bash
python tools/train.py \
    <config> \
    [--work-dir WORK_DIR] \
    [--resume] \
    [--amp] \
    [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]] \
    [--launcher {none,pytorch,slurm,mpi}] \
    [--local_rank LOCAL_RANK]
```

**Positional arguments:**
- `config`: Train config file path

**Options:**
- `--work-dir`: Directory to save logs and models
- `--resume`: Resume from the latest checkpoint in the work_dir automatically
- `--amp`: Enable automatic-mixed-precision training
- `--cfg-options`: Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white space is allowed.
- `--launcher`: Job launcher (none, pytorch, slurm, mpi)
- `--local_rank`: Local rank for distributed training

**Example:**
```bash
python tools/train.py \
    configs/train/ycor-lm-3cls-exps/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-40k_ycor-1024x544.py \
    --work-dir work_dirs/my_experiment
```

On successful finish, a curated `work_dirs/<experiment>/summary.json` is written (val metrics, history, per-class IoU from the log, git/env). To skip, pass `--no-summarize`. To regenerate: `python tools/summarize_experiment.py <work_dir>`. See `docs/mlops.md`.

### Inference

Run inference on images, videos, or live streams.

```bash
python tools/inference.py \
    --input <input_path> \
    --cfg-uri <config_path> \
    --model-uri <model_path> \
    [--backend {torch,onnx,tensorrt}] \
    [--save] \
    [--show]
```

**Options:**
- `--input`, `-i`: Path to image, video, directory, camera ID, or stream URL.
- `--cfg-uri`, `-c`: Path to model config (.py for torch, pipeline.json for engines).
- `--model-uri`, `-m`: Path to model weights (.pth, .onnx, or .engine).
- `--backend`, `-b`: Inference backend (default: `torch`).
- `--device`: Device used for inference (default: `cuda:0`).
- `--output-dir`, `-o`: Root directory to save results (default: `work_dirs/inference`).
- `--save`: Master flag to enable saving results to disk.
- `--save-vis`: Save visualized overlay results (default if `--save` is used).
- `--save-mask`: Save raw 1-channel segmentation masks (.png).
- `--show`: Show results in a real-time window.
- `--overlay-fps`: Draw real-time FPS on the results.
- `--conf-threshold`: Confidence threshold (single float or per-class list).
- `--batch-size`: Number of frames to process in a single batch.
- `--opacity`: Alpha opacity for visualization overlay (default: `0.7`).

**Examples:**

```bash
# 1. Single image with live display
python tools/inference.py -i demo.jpg -c configs/model.py -m model.pth --show

# 2. Video file with TensorRT backend and saving results
python tools/inference.py -i video.mp4 -c pipeline.json -m model.engine -b tensorrt --save --overlay-fps

# 3. Directory processing with raw mask saving
python tools/inference.py -i data/test_imgs/ -c configs/model.py -m model.pth --save --save-mask
```

### Deploy

Export PyTorch model to ONNX (and other backends). For ONNX Runtime, the script can optionally **rewrite mmdeploy/mmcv custom ops to standard ONNX** in place (e.g. `grid_sampler` → `GridSample`), so the output runs without mmdeploy’s custom op library. Rewrite is in-memory then one write to `end2end.onnx`; use `--no-rewrite` to keep custom ops.

```bash
python tools/deploy/deploy.py <deploy_cfg> <model_cfg> <checkpoint> <img> [options]
```

**Positional arguments:**
- `deploy_cfg`: Deploy config path (e.g. `configs/deploy/custom/segmentation_onnxruntime_dynamic.py`)
- `model_cfg`: Model config path (e.g. `work_dirs/my_exp/config.py`)
- `checkpoint`: Model checkpoint path (e.g. `work_dirs/my_exp/best.pth`)
- `img`: Image used for conversion (e.g. `assets/image/sample.jpg`)

**Options:**
- `--work-dir`: Directory to save logs and exported model (default: current dir). Output ONNX: `end2end.onnx`.
- `--device`: Device for conversion (default: `cpu`). Use `cuda` for GPU.
- `--dump-info`: Output SDK metadata (pipeline.json, etc.).
- `--no-rewrite`: Skip rewriting custom ops; keep mmdeploy custom ops (requires mmdeploy runtime at inference).
- `--show`: Run visualization after export.
- `--test-img`: Image(s) for testing the exported model.
- `--log-level`: Log level (default: `INFO`).
- `--calib-dataset-cfg`, `--quant-image-dir`, `--quant`, `--uri`: Calibration and quantization options (see script help).

**Examples:**

```bash
# Export to ONNX and rewrite custom ops → work_dir/end2end.onnx (standard ONNX)
python tools/deploy/deploy.py \
    configs/deploy/custom/segmentation_onnxruntime_dynamic.py \
    work_dirs/my_exp/config.py \
    work_dirs/my_exp/best.pth \
    assets/image/sample.jpg \
    --work-dir mmdeploy_model/onnx \
    --device cuda \
    --dump-info

# Export only, no rewrite (keep custom ops)
python tools/deploy/deploy.py ... --no-rewrite

# Export and show visualization
python tools/deploy/deploy.py ... --show

# Mask2Former — use the dedicated deploy config (avoids traced-tensor deepcopy crash)
python tools/deploy/deploy.py \
    configs/deploy/custom/segmentation_onnxruntime_dynamic_mask2former.py \
    configs/train/.../mask2former_r50_8xb2-90k_ycor-1024x544.py \
    work_dirs/my_exp/best.pth \
    assets/image/sample.jpg \
    --work-dir work_dirs/my_exp/deploy/onnx
```

> **Mask2Former note:** using the generic `segmentation_onnxruntime_dynamic.py` config with a
> Mask2Former checkpoint raises `RuntimeError: NYI: Named tensors are not supported with the tracer`.
> See [docs/deploy.md](docs/deploy.md) for the full explanation and fix.

## Project Structure

```
├── configs/          # Model and dataset configurations
├── docs/             # Design docs and deployment notes
│   ├── mlops.md      #   End-to-end MLOps workflow (train → deploy → release)
│   └── deploy.md     #   ONNX export guide and known issues
├── src/              # Custom datasets, models, and utilities
├── tools/            # Training and inference scripts
├── data/             # Dataset directory (excluded from git)
└── work_dirs/        # Training outputs and checkpoints (excluded from git)
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.13.0
- MMSegmentation >= 1.0.0
- See `requirements.txt` for full list

## Changelog (configs)

- Training configs use `type='FixedCrossEntropyLoss'` (see `src/mowing_terrain_seg/models/losses/cross_entropy_loss.py`) instead of monkey-patching MMSeg’s `cross_entropy`; older `work_dirs/` runs may reference `CrossEntropyLoss` in saved `config.py` copies.

## License

See LICENSE file for details.

