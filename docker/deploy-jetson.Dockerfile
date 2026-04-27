# Jetson: TensorRT engine build (trtexec) + tools/build_engine.py + tools/release.py --engine-dir
# Build on an aarch64 host with the base image, or: docker buildx build --platform linux/arm64 .
FROM ghcr.io/serene4mr/ros:humble-desktop-l4t-r36.4.0-fix

COPY requirements-deploy.txt /tmp/requirements-deploy.txt
RUN pip install --no-cache-dir -r /tmp/requirements-deploy.txt

COPY pyproject.toml /workspace/pyproject.toml
COPY tools/__init__.py /workspace/tools/__init__.py
COPY tools/build_engine.py /workspace/tools/build_engine.py
COPY tools/release.py /workspace/tools/release.py
COPY tools/hf_release /workspace/tools/hf_release
COPY tools/summarize_experiment.py /workspace/tools/summarize_experiment.py

# Match host TensorRT; common Jetson L4T layout
ENV PATH="/usr/src/tensorrt/bin:${PATH}"
WORKDIR /workspace
ENV PYTHONPATH=/workspace
# Example:
#   docker run --rm -it --runtime nvidia -v $(pwd):/workspace mts-deploy-jetson \\
#     python3 tools/build_engine.py --repo-id org/m --revision v1 --output-dir /tmp/e
#   python3 tools/release.py --engine-dir /tmp/e --repo-id org/m --tag v1 --dry-run
