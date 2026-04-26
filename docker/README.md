# Docker Image Build Guide

This directory contains the base environment image for training, testing, and deployment.

## Files

- `Dockerfile`: Base image with CUDA, PyTorch, ONNX Runtime, and Python dependencies.
- `build.sh`: Helper script to build and optionally push the image to GHCR.

## Build

Run:

```bash
bash docker/build.sh
```

Default output image:

- `ghcr.io/serene/mowing-terrain-seg:base`

## Supported args

- `--tag <tag>`: Override image tag (default: `base`)
- `--push`: Push the built image to GHCR
- `--no-cache`: Disable Docker build cache

Examples:

```bash
bash docker/build.sh --tag v1.0
bash docker/build.sh --tag v1.0 --no-cache
bash docker/build.sh --tag v1.0 --push
```

## Notes

- `build.sh` always uses repo root as build context so `requirements.txt` is available to `Dockerfile`.
- `.devcontainer/Dockerfile` is a separate dev-only layer that creates a non-root user on top of this base image.
- Log in before pushing:

```bash
docker login ghcr.io
```
