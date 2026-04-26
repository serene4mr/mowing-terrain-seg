"""Confidence thresholding."""
import numpy as np

from mowing_terrain_seg.inference.thresholding import apply_confidence_threshold


def test_single_threshold_filters_low_confidence():
    mask = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    conf = np.array([[0.9, 0.2], [0.8, 0.8]], dtype=np.float32)
    out = apply_confidence_threshold(
        mask, conf, 0.5, 2, None
    )
    assert out[0, 0] == 0
    assert out[0, 1] == 255
    assert out[1, 0] == 1
    assert out[1, 1] == 1


def test_per_class_thresholds():
    mask = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    conf = np.array([[0.3, 0.9], [0.8, 0.4]], dtype=np.float32)
    out = apply_confidence_threshold(
        mask, conf, [0.5, 0.5], 2, None
    )
    # (0,0) class 0, conf 0.3 < 0.5 -> 255
    assert out[0, 0] == 255
    # (0,1) class 1, conf 0.9 >= 0.5
    assert out[0, 1] == 1
    # (1,0) class 0, conf 0.8 >= 0.5
    assert out[1, 0] == 0
    # (1,1) class 1, conf 0.4 < 0.5
    assert out[1, 1] == 255
