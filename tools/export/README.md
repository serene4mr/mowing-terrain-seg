# Model Export & Optimization Tools

This directory contains scripts for converting trained models from PyTorch to deployment-ready formats (ONNX, TensorRT).

## 1. PyTorch to ONNX (`torch2onnx.py`)

Exports a trained model checkpoint to ONNX format using **MMDeploy**. This script also handles custom op rewriting to ensure standard ONNX compatibility.

**Usage:**
```bash
python tools/export/torch2onnx.py \
    configs/deploy/custom/segmentation_onnxruntime_dynamic.py \
    configs/train/segformer_mit-b0_8xb2-160k_mowing-terrain.py \
    work_dirs/experiment_name/best_mIoU_iter_1000.pth \
    data/sample_image.jpg \
    --work-dir work_dirs/export \
    --device cuda
```

---

## 2. ONNX Post-Processing (`onnx_add_argmax_output.py`)

Appends an `ArgMax` node to an existing ONNX model. Use this when the base model outputs raw logits (digits) but you also want the final predicted class indices in a single inference pass.

**Usage:**
```bash
python tools/export/onnx_add_argmax_output.py \
    work_dirs/export/end2end.onnx \
    work_dirs/export/end2end_dual.onnx \
    --axis 1
```
*   `--axis`: The dimension to reduce (default: 1 for NCHW).
*   `--drop-dims`: If set, drops the channel dimension (output shape becomes `[N, H, W]` instead of `[N, 1, H, W]`).

---

## 3. ONNX to TensorRT (`onnx2tensorrt.py`)

Converts an ONNX model into a highly optimized TensorRT engine for NVIDIA edge devices (e.g., Jetson).

**Usage:**
```bash
python tools/export/onnx2tensorrt.py \
    configs/deploy/mmseg/segmentation_tensorrt_static-640x480.py \
    work_dirs/export \
    --model work_dirs/export/end2end.onnx \
    --device cuda:0
```

---

## Prerequisites

Ensure you have the required dependencies installed:
```bash
pip install onnx onnxruntime-gpu mmdeploy==1.3.1
```
*(Note: ONNX Runtime installation might vary depending on your CUDA version and environment; see `requirements.txt` for details).*
