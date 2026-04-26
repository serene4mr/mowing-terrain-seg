"""CustomSegLocalVisualizer confidence map."""
import tempfile

import numpy as np
import torch
from mmengine.structures import PixelData

from mowing_terrain_seg.visualization.custom_local_visualizer import (
    CustomSegLocalVisualizer,
)


def test_draw_conf_map_shape_and_range():
    with tempfile.TemporaryDirectory() as d:
        v = CustomSegLocalVisualizer(
            save_dir=d,
        )
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    logits = torch.randn(2, 8, 8)
    pd = PixelData()
    pd.data = logits
    out = v._draw_conf_map(img, pd)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert out.max() > 0
