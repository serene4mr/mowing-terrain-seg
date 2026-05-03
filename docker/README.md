# Docker Environments

This directory contains the Dockerfiles and helper scripts to build the training and deployment environments for the `mowing-terrain-seg` project.

## Files

- `Dockerfile`: The base development image with CUDA, PyTorch, OpenMMLab, ONNX Runtime, and MMDeploy. Suitable for x86 dev machines.
- `deploy-jetson.Dockerfile`: A lean deployment image targeting NVIDIA Jetson devices (aarch64). Contains TensorRT build tools (`trtexec`) and deployment scripts.
- `build.sh`: A helper script to build and optionally push the base development image to GitHub Container Registry (GHCR).
- `verify_env.py`: A script to verify that the development environment is set up correctly (checks PyTorch, OpenMMLab versions, CUDA paths, and optionally runs inference).

---

## 1. Building the Development Image

Run the build script from the repository root:

```bash
bash docker/build.sh
```

**Default output image:**
- `ghcr.io/serene/mowing-terrain-seg:base`

**Supported flags:**
- `--tag <tag>`: Override image tag (default: `base`)
- `--push`: Push the built image to GHCR
- `--no-cache`: Disable Docker build cache

**Examples:**
```bash
bash docker/build.sh --tag v1.0
bash docker/build.sh --tag v1.0 --no-cache
bash docker/build.sh --tag v1.0 --push
```

### Verifying the Environment
Once inside the development container, you can verify your stack:
```bash
python3 docker/verify_env.py
```
*Pass `--no-inference` if you do not have an exported model to test yet.*

---

## 2. Building the Jetson Deployment Image

The Jetson image (`deploy-jetson.Dockerfile`) must be built on an `aarch64` host or using cross-compilation on your x86 machine. Run this from the repository root:

**On a Jetson or ARM64 host:**
```bash
docker build -t mts-deploy-jetson -f docker/deploy-jetson.Dockerfile .
```

**On an x86 host (Cross-compilation):**
```bash
docker buildx build --platform linux/arm64 -t mts-deploy-jetson -f docker/deploy-jetson.Dockerfile . --load
```

For more details on how to use this image for building TensorRT engines, see the [Deploy Guide](../docs/deploy.md).
