"""Custom classes registered in MMSeg / MMEngine registries."""
import mmseg

from mmseg.registry import DATASETS, MODELS, VISUALIZERS
# Import project package so modules register
import mowing_terrain_seg  # noqa: F401
from mowing_terrain_seg.models.losses.cross_entropy_loss import (  # noqa: F401
    FixedCrossEntropyLoss,
)


def test_mmseg_version():
    assert str(mmseg.__version__).startswith("1.2.")


def test_datasets_registry():
    mowing_terrain_seg.register_all()
    assert DATASETS.get("YCORDataset") is not None
    assert DATASETS.get("YCORLawnMowing3ClassDataset") is not None


def test_loss_registry():
    assert MODELS.get("FixedCrossEntropyLoss") is not None


def test_visualizer_registry():
    assert VISUALIZERS.get("CustomSegLocalVisualizer") is not None
