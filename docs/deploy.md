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
