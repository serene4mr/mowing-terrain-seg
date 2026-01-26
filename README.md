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

```bash
pip install -r requirements.txt
```

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

Export model to backends.

```bash
python tools/deploy_tools/deploy.py \
    <deploy_cfg> \
    <model_cfg> \
    <checkpoint> \
    <img> \
    [--test-img TEST_IMG [TEST_IMG ...]] \
    [--work-dir WORK_DIR] \
    [--calib-dataset-cfg CALIB_DATASET_CFG] \
    [--device DEVICE] \
    [--log-level {CRITICAL,FATAL,ERROR,WARN,WARNING,INFO,DEBUG,NOTSET}] \
    [--show] \
    [--dump-info] \
    [--quant-image-dir QUANT_IMAGE_DIR] \
    [--quant] \
    [--uri URI]
```

**Positional arguments:**
- `deploy_cfg`: Deploy config path
- `model_cfg`: Model config path
- `checkpoint`: Model checkpoint path
- `img`: Image used to convert model

**Options:**
- `--test-img`: Image(s) used to test model
- `--work-dir`: Directory to save logs and models
- `--calib-dataset-cfg`: Dataset config path used to calibrate in int8 mode (defaults to "val" dataset in model config if not specified)
- `--device`: Device used for conversion
- `--log-level`: Set log level
- `--show`: Show detection outputs
- `--dump-info`: Output information for SDK
- `--quant-image-dir`: Image directory for quantize model
- `--quant`: Quantize model to low bit
- `--uri`: Remote ipv4:port or ipv6:port for inference on edge device

## Project Structure

```
├── configs/          # Model and dataset configurations
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

## License

See LICENSE file for details.

