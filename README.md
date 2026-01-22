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

```bash
python tools/train.py \
    configs/train/ycor-lm-3cls-exps/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-40k_ycor-1024x544.py \
    --work-dir work_dirs/my_experiment
```

### Inference

```bash
python tools/inference.py \
    --input path/to/image.jpg \
    --config configs/model.py \
    --checkpoint work_dirs/model.pth \
    --output-dir results/
```

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

