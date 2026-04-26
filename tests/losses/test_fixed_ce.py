"""FixedCrossEntropyLoss vs upstream CE with class weights + ignore_index."""
import torch

from mmseg.registry import MODELS

from mowing_terrain_seg.models.losses.cross_entropy_loss import (
    FixedCrossEntropyLoss,
    fixed_cross_entropy,
)


def test_registry_has_fixed_ce():
    cls = MODELS.get("FixedCrossEntropyLoss")
    assert cls is not None
    assert cls is FixedCrossEntropyLoss


def test_fixed_matches_standalone_function():
    pred = torch.randn(2, 3, 4, 4)
    label = torch.randint(0, 3, (2, 4, 4))
    label[:, 0, :] = 255
    w = torch.tensor([1.0, 2.0, 1.0])
    a = fixed_cross_entropy(
        pred,
        label,
        class_weight=w,
        reduction="mean",
        ignore_index=255,
        avg_non_ignore=True,
    )
    mod = FixedCrossEntropyLoss(
        class_weight=[1.0, 2.0, 1.0],
        avg_non_ignore=True,
        use_sigmoid=False,
    )
    b = mod(pred, label, ignore_index=255)
    assert torch.allclose(a, b)


def test_fixed_handles_ignore_index_with_class_weights():
    """``fixed_cross_entropy`` must not index ``class_weight`` at ignore_index (e.g. 255)."""
    pred = torch.randn(1, 3, 4, 4)
    label = torch.zeros(1, 4, 4, dtype=torch.long)
    label[:, :, 0] = 255
    w = torch.tensor([1.0, 2.0, 3.0])
    fx = fixed_cross_entropy(
        pred,
        label,
        class_weight=w,
        reduction="mean",
        ignore_index=255,
        avg_non_ignore=True,
    )
    assert torch.isfinite(fx)


def test_build_from_config_dict():
    cfg = dict(
        type="FixedCrossEntropyLoss",
        use_sigmoid=False,
        loss_weight=1.0,
        class_weight=[1.0, 1.0, 1.0],
        avg_non_ignore=True,
    )
    loss = MODELS.build(cfg)
    assert isinstance(loss, FixedCrossEntropyLoss)
    pred = torch.randn(1, 3, 2, 2, requires_grad=True)
    label = torch.zeros(1, 2, 2, dtype=torch.long)
    out = loss(pred, label, ignore_index=255)
    out.backward()
    assert pred.grad is not None
