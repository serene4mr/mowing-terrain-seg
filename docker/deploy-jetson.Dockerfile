# Jetson: TensorRT engine build (trtexec) + ONNX post-processing + HF release tools.
# Build on an aarch64 host with the base image, or: docker buildx build --platform linux/arm64 .
# FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ghcr.io/serene4mr/mowbot:devel-jetson-l4t-r36.4-latest

# Install pip and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt /tmp/requirements-deploy.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r /tmp/requirements-deploy.txt

COPY pyproject.toml /workspace/pyproject.toml
COPY tools/__init__.py /workspace/tools/__init__.py
COPY tools/export/onnx2tensorrt.py /workspace/tools/export/onnx2tensorrt.py
COPY tools/export/onnx_add_argmax_output.py /workspace/tools/export/onnx_add_argmax_output.py
COPY tools/push_hf_repo.py /workspace/tools/push_hf_repo.py

# Match host TensorRT; common Jetson L4T layout
ENV PATH="/usr/src/tensorrt/bin:${PATH}"
WORKDIR /workspace
ENV PYTHONPATH=/workspace

# Usage Example:
#
#   docker run --rm -it --runtime nvidia -v $(pwd):/workspace mts-deploy-jetson \
#     python3 tools/export/onnx2tensorrt.py --onnx deploy/onnx/end2end_dual.onnx --output-dir deploy/tensorrt
#
#   docker run --rm -it -v $(pwd):/workspace mts-deploy-jetson \
#     python3 tools/push_hf_repo.py --local-dir deploy/tensorrt --repo-id org/mowing-terrain-seg
