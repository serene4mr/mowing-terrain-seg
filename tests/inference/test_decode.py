"""Tests for decode helpers."""
import numpy as np
import torch

from mowing_terrain_seg.inference.decode import (
    extract_mask_unified,
    compute_confidence_unified,
)


def test_onnx_logits_to_mask_and_confidence():
    # (1, 3, 2, 2) logits — class 2 wins at (0,0)
    logits = np.zeros((1, 3, 2, 2), dtype=np.float32)
    logits[0, 2, 0, 0] = 5.0
    mask = extract_mask_unified(logits)
    assert mask.shape == (2, 2)
    assert mask[0, 0] == 2
    conf = compute_confidence_unified(logits)
    assert conf is not None
    assert conf.shape == (2, 2)
    assert conf[0, 0] > 0.9


def test_onnx_prediction_map_no_confidence():
    pred = np.zeros((1, 1, 2, 2), dtype=np.uint8)
    pred[0, 0, 0, 0] = 1
    mask = extract_mask_unified(pred)
    assert mask[0, 0] == 1
    assert compute_confidence_unified(pred) is None


def test_fake_data_sample_like_torch():
    class _SegLogits:
        def __init__(self, t):
            self.data = t

    class _Pred:
        def __init__(self):
            self.data = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)

    class _Sample:
        def __init__(self):
            self.pred_sem_seg = _Pred()
            self.seg_logits = _SegLogits(
                torch.randn(3, 2, 2)
            )

    s = _Sample()
    m = extract_mask_unified(s)
    assert m.shape == (2, 2)
    c = compute_confidence_unified(s)
    assert c is not None and c.shape == (2, 2)
