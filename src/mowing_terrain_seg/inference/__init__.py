import time
import torch
import numpy as np

from .predictor import SegPredictor, Backend
from .source import InferenceSource, SourceType

class InferenceTimer:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.is_cuda = 'cuda' in device
        self.pre_times = []
        self.infer_times = []
        self.post_times = []
        self.total_times = []
        self.start_tick = 0

    def synchronize(self):
        if self.is_cuda:
            torch.cuda.synchronize()

    def tick(self):
        self.synchronize()
        return time.time()

    def record(self, pre, infer, post):
        self.pre_times.append(pre)
        self.infer_times.append(infer)
        self.post_times.append(post)
        self.total_times.append(pre + infer + post)

    def get_avg_fps(self):
        """Cumulative average FPS over all recorded frames."""
        if not self.total_times:
            return 0
        return 1.0 / (sum(self.total_times) / len(self.total_times))

    def get_sliding_fps(self, window=30):
        """FPS over the last `window` frames. Use for live overlay (stable, no drift)."""
        if not self.total_times:
            return 0
        recent = self.total_times[-window:]
        avg_latency = sum(recent) / len(recent)
        return 1.0 / avg_latency if avg_latency > 0 else 0

    def get_stats(self, warmup_frames=30):
        """
        Stats over frames after warm-up. If total frames <= warmup_frames, uses all frames.
        Use for final report (steady-state, comparable across runs).
        """
        if not self.total_times:
            return {}
        n = len(self.total_times)
        skip = min(warmup_frames, n - 1) if n > 1 else 0
        pre = self.pre_times[skip:]
        infer = self.infer_times[skip:]
        post = self.post_times[skip:]
        total = self.total_times[skip:]
        n_used = len(total)
        if n_used == 0:
            return {}
        return {
            'avg_pre': sum(pre) / n_used * 1000,
            'avg_infer': sum(infer) / n_used * 1000,
            'avg_post': sum(post) / n_used * 1000,
            'avg_total': sum(total) / n_used * 1000,
            'avg_fps': 1.0 / (sum(total) / n_used),
            'p99_latency': np.percentile(total, 99) * 1000,
            'warmup_frames': skip,
            'frames_used': n_used,
        }
