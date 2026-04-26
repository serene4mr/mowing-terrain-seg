"""YCOR 8-class -> 3-class mapping and lookup tables (see ycor.py docstring)."""
import os
import tempfile

import pytest
import torch
from PIL import Image

from mowing_terrain_seg.datasets.ycor import YCORLawnMowing3ClassDataset

# Documented mapping from YCORLawnMowing3ClassDataset docstring
EXPECTED_MAP = {
    0: 2,  # background -> Non-traversable
    1: 1,  # smooth_trail -> Traversable
    2: 0,  # traversable_grass -> Cuttable
    3: 1,  # rough_trail -> Traversable
    4: 2,  # puddle -> Non-traversable
    5: 2,  # obstacle -> Non-traversable
    6: 0,  # non_traversable_vegetation -> Cuttable
    7: 2,  # high_vegetation -> Non-traversable
    8: 2,  # sky -> Non-traversable
}


@pytest.fixture
def minimal_ycor_root():
    with tempfile.TemporaryDirectory() as root:
        train = os.path.join(root, "train", "iid000001")
        os.makedirs(train)
        Image.new("RGB", (32, 32), color=(128, 128, 128)).save(
            os.path.join(train, "rgb.jpg")
        )
        Image.new("L", (32, 32), color=0).save(
            os.path.join(train, "labels.png")
        )
        yield root


def test_label_map_matches_documentation(minimal_ycor_root):
    ds = YCORLawnMowing3ClassDataset(
        data_root=minimal_ycor_root,
        data_prefix=dict(img_path="train", seg_map_path="train"),
    )
    for k, v in EXPECTED_MAP.items():
        assert ds.label_map[k] == v


def test_lookup_table_indices_0_to_8(minimal_ycor_root):
    ds = YCORLawnMowing3ClassDataset(
        data_root=minimal_ycor_root,
        data_prefix=dict(img_path="train", seg_map_path="train"),
    )
    for old_id, new_id in EXPECTED_MAP.items():
        assert ds.lookup_table[old_id] == new_id
    assert ds.lookup_table_t.shape == (256,)
    assert (ds.lookup_table_t.numpy() == ds.lookup_table).all()


def test_remap_tensor_on_device(minimal_ycor_root):
    ds = YCORLawnMowing3ClassDataset(
        data_root=minimal_ycor_root,
        data_prefix=dict(img_path="train", seg_map_path="train"),
    )
    x = torch.tensor([[0, 1, 2], [3, 8, 255]], dtype=torch.long)
    out = ds._remap_tensor(x)
    assert out.shape == x.shape
    assert out.device == x.device
    assert out[0, 0].item() == 2
    assert out[0, 1].item() == 1
    assert out[0, 2].item() == 0
    assert out[1, 0].item() == 1
    assert out[1, 1].item() == 2
    assert out[1, 2].item() == ds.lookup_table[255]
