# Jetson: TensorRT engine build (trtexec) + tools/build_engine.py + tools/release.py --engine-dir
# Build on an aarch64 host with the base image, or: docker buildx build --platform linux/arm64 .
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

COPY requirements-deploy.txt /tmp/requirements-deploy.txt
RUN pip install --no-cache-dir -r /tmp/requirements-deploy.txt

COPY pyproject.toml /workspace/pyproject.toml
COPY tools/__init__.py /workspace/tools/__init__.py
COPY tools/export/onnx2tensorrt.py /workspace/tools/export/onnx2tensorrt.py
COPY tools/push_hf_repo.py /workspace/tools/push_hf_repo.py

# Match host TensorRT; common Jetson L4T layout
ENV PATH="/usr/src/tensorrt/bin:${PATH}"
WORKDIR /workspace
ENV PYTHONPATH=/workspace
# Example:
#   docker run --rm -it --runtime nvidia -v $(pwd):/workspace mts-deploy-jetson \\
#     python3 tools/export/onnx2tensorrt.py --onnx deploy/onnx/end2end.onnx --output-dir deploy/tensorrt
#   python3 tools/push_hf_repo.py --local-dir deploy/tensorrt --repo-id org/m
