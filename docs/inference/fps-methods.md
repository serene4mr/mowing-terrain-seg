# FPS Methods Used in Inference

This document describes how FPS (frames per second) and latency are measured and reported in the segmentation inference pipeline (`tools/inference.py` and `InferenceTimer` in `src/mowing_terrain_seg/inference/`).

## What Is Timed (Per Frame)

For each frame (or batch of images), we record:

| Phase | Description | Included in |
|-------|-------------|-------------|
| **Infer** | Preprocess + model forward + model postprocess (e.g. argmax) | `predictor.predict(imgs)` |
| **Post** | Visualization and I/O (e.g. overlay, save) | Loop over `visualize_mask`, `imwrite`, etc. |

**Pre** is not measured separately (passed as 0). The **total time per frame** is:

```
total = infer + post   (seconds)
```

All timings use `torch.cuda.synchronize()` when running on CUDA so that GPU work is included before reading the clock.

---

## FPS Methods

FPS is always defined as the inverse of average latency (in seconds):

```
FPS = 1 / average_latency
```

We use three ways to compute that average, depending on the use case.

### 1. Cumulative Average (`get_avg_fps`)

- **Formula:** `FPS = 1 / (sum(total_times) / len(total_times))`
- **Scope:** All frames from the start of the run.
- **Use:** Available in the API; not used for overlay or final report in the current pipeline.
- **Behavior:** As more frames are processed, the average includes more “warmed up” frames, so FPS tends to **increase over time** (early frames are slower). Good for a single “whole run” number; not ideal for a stable on-screen value or for comparing steady-state performance across runs.

### 2. Sliding-Window FPS (`get_sliding_fps`)

- **Formula:** `FPS = 1 / (sum(last N total_times) / N)` with default `N = 30`.
- **Scope:** Only the last 30 recorded frames.
- **Use:** **Live overlay** when `--overlay-fps` is set (FPS drawn on each frame).
- **Rationale:** The on-screen value stabilizes quickly and reflects *current* throughput without drifting upward. If fewer than 30 frames have been recorded, the average uses all available frames.

### 3. Warm-Up Skip in Final Stats (`get_stats(warmup_frames=30)`)

- **Formula:** Drop the first 30 frames, then compute averages and P99 over the remaining frames:  
  `FPS = 1 / (sum(total_times[30:]) / len(total_times[30:]))`, same for pre/infer/post and P99.
- **Scope:** All frames *after* the first 30 (or all frames if total ≤ 30).
- **Use:** **Final performance report** (printed summary and `performance.json`).
- **Rationale:** Early frames are often slower (CUDA/ONNX warm-up, JIT, etc.). Reporting stats after warm-up gives a **steady-state**, comparable number across runs. If there are fewer than 30 frames, warm-up is reduced so that at least one frame is still used for the stats.

---

## Where Each Method Is Used

| Context | Method | Purpose |
|--------|--------|--------|
| Overlay (e.g. “FPS: 12.3” on video) | Sliding-window (last 30 frames) | Stable, recent throughput on screen |
| Final report & `performance.json` | Warm-up skip (first 30 frames) then average | Steady-state FPS and latencies for benchmarking |

The report also includes `warmup_frames` and `frames_used` in the stats so you know how many frames were skipped and how many were used for the averages.

---

## Summary

- **Overlay FPS:** Sliding window (30 frames) → stable, no long-term drift.
- **Reported “Avg FPS” and latencies:** After 30-frame warm-up → steady-state, comparable across runs.

Both use the same per-frame total time (infer + post in seconds) and the same relation `FPS = 1 / average_latency`.
