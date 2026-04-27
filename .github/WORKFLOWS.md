# GitHub Workflows

This folder contains CI workflows used as an automated gate for pull requests and pushes.

## Current CI (`workflows/ci.yml`)

- Runs on pushes to `main`/`master` and on pull requests.
- Uses Python 3.10.
- Installs CPU PyTorch + OpenMMLab stack.
- Installs this repo in editable mode with dev dependencies.
- Runs:
  - `ruff check src tools tests` (currently non-blocking)
  - `python -m pytest tests/ -q` (blocking)

## Purpose

- Catch install/import regressions on a clean machine.
- Catch test regressions before merge.
- Keep `main` healthy with repeatable checks.

## Scope limits

Current CI is CPU-only. It does **not** validate:

- CUDA/GPU runtime behavior
- TensorRT runtime behavior
- Full mmdeploy GPU deployment path

For deployment confidence, add a separate GPU workflow (self-hosted runner).
