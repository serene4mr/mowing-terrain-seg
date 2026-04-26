# ─────────────────────────────────────────────────────────────────────────────
# mowing-terrain-seg  –  base image
#   docker build -t mowing-seg:base -f docker/Dockerfile .
#   (run from repo root so requirements.txt is in build context)
#
# For dev with a non-root user, use .devcontainer/Dockerfile (FROM this image).
#
# Stack:
#   PyTorch  2.1.2  +  CUDA 11.8  +  cuDNN 8
#   MMDeploy 1.3.1  +  ONNX Runtime 1.15.1 (C++ SDK tarball, NOT pip wheel)
#
# Why the C++ SDK tarball instead of pip install onnxruntime-gpu?
#   mmdeploy-runtime-gpu is a compiled C++ SDK. The official MMDeploy docs
#   prescribe installing the ORT C++ SDK tarball and setting ONNXRUNTIME_DIR /
#   LD_LIBRARY_PATH. The pip wheel has a VERNEED/NEEDED mismatch against glibc
#   2.31 (Ubuntu 20.04) that causes a dl-version.c assertion at dlopen time.
#   The tarball build does not have this issue.
#   Ref: https://mmdeploy.readthedocs.io/en/1.x/get_started.html
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        git \
        libgl1 \
        libglib2.0-0 \
        libsm6 libice6 libxrender1 libxext6 libxcb-xinerama0 \
    && rm -rf /var/lib/apt/lists/*

# ── ONNX Runtime C++ SDK (official tarball) ───────────────────────────────────
# This is the approach prescribed by the MMDeploy docs. The tarball ships
# libonnxruntime.so, libonnxruntime_providers_cuda.so, _providers_shared.so,
# and _providers_tensorrt.so — all built and tested as a unit, no glibc issues.
ARG ORT_VERSION=1.15.1
RUN wget -q \
    "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz" \
    -O /tmp/ort.tgz \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm /tmp/ort.tgz

ENV ONNXRUNTIME_DIR=/opt/onnxruntime
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:/usr/local/cuda-11.8/lib64:/usr/lib/x86_64-linux-gnu

# ── Python packages ───────────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Editable install of this repo (import `mowing_terrain_seg` without PYTHONPATH)
COPY pyproject.toml README.md /tmp/mts/
COPY src /tmp/mts/src
RUN pip install --no-cache-dir -e /tmp/mts
