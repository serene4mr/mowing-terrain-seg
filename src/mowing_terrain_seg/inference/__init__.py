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
        if not self.total_times:
            return 0
        return 1.0 / (sum(self.total_times) / len(self.total_times))

    def get_stats(self):
        if not self.total_times:
            return {}
        n = len(self.total_times)
        return {
            'avg_pre': sum(self.pre_times) / n * 1000,
            'avg_infer': sum(self.infer_times) / n * 1000,
            'avg_post': sum(self.post_times) / n * 1000,
            'avg_total': sum(self.total_times) / n * 1000,
            'avg_fps': self.get_avg_fps(),
            'p99_latency': np.percentile(self.total_times, 99) * 1000
        }
