# Deployment guide

This document covers how to export trained models to ONNX and known issues
encountered in the process.

See [mlops.md](mlops.md) for the end-to-end MLOps workflow (train → deploy →
release).

---

## Quick reference

```bash
# Generic segmentation model (DeepLabV3+, etc.)
python tools/deploy/deploy.py \
  configs/deploy/custom/segmentation_onnxruntime_dynamic.py \
  <model_cfg.py> <checkpoint.pth> <sample_image.jpg> \
  --work-dir <output_dir>

# Mask2Former  ← use the dedicated deploy config (see §2 below)
python tools/deploy/deploy.py \
  configs/deploy/custom/segmentation_onnxruntime_dynamic_mask2former.py \
  <model_cfg.py> <checkpoint.pth> <sample_image.jpg> \
  --work-dir <output_dir>
```

---

## Deploy configs

| Config file | Use for |
|---|---|
| `segmentation_onnxruntime_dynamic.py` | Generic mmseg models (DeepLabV3+, SegFormer, …) |
| `segmentation_onnxruntime_static-1024x544.py` | Any model, fixed 1024×544 input |
| `segmentation_onnxruntime_dynamic_mask2former.py` | **Mask2Former** (loads a custom tracer fix, see §2) |

---

## Known issues

### 1 — `RuntimeError: NYI: Named tensors are not supported with the tracer`

**Affects:** Mask2Former (mmseg) exported with a *dynamic-shape* deploy config
via mmdeploy.

**Symptom**

```
RuntimeError: NYI: Named tensors are not supported with the tracer
```

Full call stack (condensed):

```
torch.onnx.export
  → mmdeploy encoder_decoder__predict          # mmdeploy rewriter
  → mmseg Mask2FormerHead.predict (line 146)
      batch_data_samples = [
          SegDataSample(metainfo=metainfo)      # ← crashes here
          for metainfo in batch_img_metas
      ]
  → BaseDataElement.set_metainfo
  → copy.deepcopy(metainfo_dict)
  → Tensor.__deepcopy__ → storage.clone()
  → RuntimeError
```

**Root cause**

mmdeploy's `encoder_decoder__predict` rewriter passes `batch_img_metas` as
plain Python dicts to the decode head.  Under *dynamic-shape* export the
values in those dicts (e.g. `img_shape`) are **traced `torch.Tensor`
objects**, not plain integers.

`Mask2FormerHead.predict` reconstructs `SegDataSample` objects by calling
`SegDataSample(metainfo=metainfo)`.  The constructor calls
`set_metainfo(metainfo)` which does `copy.deepcopy(metainfo)`.
`deepcopy` on a Tensor calls `storage.clone()`, which the JIT tracer cannot
handle, raising the error above.

mmdeploy's own `copy__default` rewriter intercepts `deepcopy(tensor)` at
the **top level**, but misses tensors **nested inside a dict**, so the
`deepcopy(metainfo_dict)` path is not guarded.

> The same underlying issue exists in mmdet's MaskFormer, where mmdeploy's
> own rewriter carries the comment:
> *"note that we can not use `set_metainfo`, deepcopy would crash the onnx
> trace"* (`mmdeploy/codebase/mmdet/models/detectors/maskformer.py`).

**Fix**

A custom `FUNCTION_REWRITER` for `Mask2FormerHead.predict` builds the
`SegDataSample` objects using `set_field` instead of `set_metainfo`.
`set_field` calls `object.__setattr__` directly — no deepcopy, tracer-safe.

```
src/mowing_terrain_seg/deploy/mask2former_rewriter.py
```

The rewriter is loaded into the export subprocess via `custom_imports` in
the dedicated deploy config:

```
configs/deploy/custom/segmentation_onnxruntime_dynamic_mask2former.py
```

`mmengine.Config.fromfile` processes `custom_imports` in every subprocess
that loads the config, so the rewriter is registered before
`torch.onnx.export` is called.

**How to use**

Replace the deploy config argument with the Mask2Former-specific one:

```bash
python tools/deploy/deploy.py \
  configs/deploy/custom/segmentation_onnxruntime_dynamic_mask2former.py \
  configs/train/ycor-lm-3cls-exps-0/mask2former/mask2former_r50_8xb2-90k_ycor-1024x544.py \
  work_dirs/<exp>/best_val_mIoU_iter_<N>.pth \
  assets/image/test1/rgb/frame_001218_rgb.png \
  --work-dir work_dirs/<exp>/deploy/onnx
```

**Files introduced**

| File | Purpose |
|---|---|
| `src/mowing_terrain_seg/deploy/mask2former_rewriter.py` | `FUNCTION_REWRITER` that replaces `SegDataSample(metainfo=…)` with `set_field` calls |
| `configs/deploy/custom/segmentation_onnxruntime_dynamic_mask2former.py` | Deploy config that inherits the dynamic config and adds `custom_imports` to load the rewriter in the subprocess |

---

## Building TensorRT engines for Jetson Orin NX

After you have published **ONNX** to Hugging Face Hub (`tools/release.py` with
`--deploy`), build a **device-specific** TensorRT engine on the Jetson itself.
Engines are **not portable** across GPU architectures; ONNX is the portable
contract.

### Prerequisites

- Jetson with TensorRT and `trtexec` (typical path: `/usr/src/tensorrt/bin/trtexec`).
- Docker: run with **`--runtime nvidia`** so TensorRT and NVDLA libraries from
  the host are visible. If `import tensorrt` fails with
  `libnvdla_compiler.so: cannot open shared object file`, you forgot
  `--runtime nvidia`.
- Hugging Face token: `HF_TOKEN` or `huggingface-cli login` (for pull/push).
- `pip install -e ".[release]"` or `pip install -e ".[deploy-trt]"` on the
  Jetson **or** use the project Dockerfile below.

### Optional: lean deploy image

From the **repository root** (so `tools/` and `requirements-deploy.txt` exist):

```bash
docker build -f docker/deploy-jetson.Dockerfile -t mts-deploy-jetson .
docker run --rm -it --runtime nvidia -v "$PWD":/workspace -w /workspace -e PYTHONPATH=/workspace mts-deploy-jetson
```

Use `python3 tools/build_engine.py ...` from `/workspace` (repo root). The
Dockerfile sets `PYTHONPATH=/workspace` for the **baked-in** copy; when you bind-mount
the live repo, pass `-e PYTHONPATH=/workspace` as above.

### 1) Build the engine (on Orin)

Pulls `onnx/*` (and a few other files) from the Hub, runs `trtexec`, writes
`end2end.engine`, `platform.json`, and `build.log`.

```bash
python tools/build_engine.py \
  --repo-id <org>/<model-repo> \
  --revision v1.0.0 \
  --output-dir work_dirs/trt/orin/v1.0.0 \
  --precision fp16
```

- **Profile** (folder name) is **auto-detected** from the board, memory, JetPack
  guess, and TensorRT Python version, unless you pass `--profile my-name`.
- **Static shapes** default to `1,3,1024,544` (`--input-shape`).
- **Local ONNX** (no Hub): `--no-pull --onnx /path/to/end2end.onnx` (still
  records provenance in `platform.json`).

`platform.json` includes `source.onnx_sha256` and `source.onnx_revision` so a
later upload can verify the engine matches the ONNX on the Hub.

**Dry run** (print `trtexec` command, no build):

```bash
python tools/build_engine.py ... --output-dir /tmp/x --dry-run
```

### 2) Upload the engine to the same Hub repo

Appends `tensorrt/<profile>/` and merges a **“Available TensorRT engines”** table
into `README.md`.

```bash
python tools/release.py \
  --engine-dir work_dirs/trt/orin/v1.0.0 \
  --repo-id <org>/<model-repo> \
  --tag v1.0.0 \
  --message "Orin TRT" \
  --allow-dirty
```

- **`--allow-dirty`**: also skips the **ONNX hash check** (Hub download vs
  `platform.json`). Omit it in CI/production so drift is caught.
- **`--profile`**: override the Hub subfolder name; `platform.json` in the
  upload is updated to match.
- Provenance is written to `hf_engine_release.json` inside `--engine-dir`.

### Hub layout

```
<repo>/
  onnx/
    end2end.onnx
    pipeline.json
    detail.json
  tensorrt/
    <auto-profile>/
      end2end.engine
      platform.json
      build.log
  README.md   # table of TensorRT profiles merged by release
```

### `platform.json` (schema v1.0, summary)

- `schema_version`, `profile`
- `device`: `board_model`, `memory_gb`, `sm_capability`, `power_mode` (when detectable)
- `software`: `l4t`, `jetpack`, `tensorrt_python`, `cuda_cudart`, `cudnn`, …
- `image` (optional): `ref` / `digest` if you pass `--docker-image-ref`
- `build`: `precision`, `workspace_mb`, `input_shape`, `trtexec_args`, `duration_sec`, `build_date`
- `source`: `onnx_repo`, `onnx_revision`, `onnx_sha256`, `onnx_path`

### Why not `tools/deploy/deploy.py` for TRT on Jetson?

`deploy.py` is mmdeploy-centric and expects a full PyTorch + mmcv stack. On
Jetson, **trtexec** (or the TensorRT API) from the L4T/JetPack image is the
lean path. ONNX from the Hub is the input; the engine is output.

---

## Adding new ONNX op rewrites (post-export)

After export, `tools/deploy/deploy.py` automatically rewrites non-standard
mmdeploy/mmcv ops to standard ONNX equivalents so the model runs on plain
ONNX Runtime without any mmdeploy runtime.

Custom op rewrites are registered in:

```
tools/deploy/_onnx_rewriter.py
```

Use `register_rewriter` to add new rules:

```python
from _onnx_rewriter import register_rewriter

@register_rewriter("mmdeploy", "my_custom_op")
def rewrite_my_op(node, graph, inputs):
    # return replacement node(s)
    ...
```

Pass `--no-rewrite` to skip post-export rewriting and keep the original
custom ops (requires mmdeploy runtime at inference time).
