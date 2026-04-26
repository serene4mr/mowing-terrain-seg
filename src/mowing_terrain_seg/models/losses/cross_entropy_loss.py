# src/mowing_terrain_seg/models/losses/cross_entropy_loss.py
import torch
import torch.nn.functional as F
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmseg.models.losses.utils import weight_reduce_loss
from mmseg.registry import MODELS


def fixed_cross_entropy(
    pred,
    label,
    weight=None,
    class_weight=None,
    reduction="mean",
    avg_factor=None,
    ignore_index=-100,
    avg_non_ignore=False,
):
    """Cross-entropy with correct ``avg_factor`` when ``class_weight`` + ``ignore_index`` are both set."""

    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction="none",
        ignore_index=ignore_index,
    )

    if (avg_factor is None) and reduction == "mean":
        if class_weight is None:
            if avg_non_ignore:
                avg_factor = label.numel() - (label == ignore_index).sum().item()
            else:
                avg_factor = label.numel()
        else:
            label_flat = label.view(-1)
            label_weights = torch.zeros_like(
                label_flat, dtype=class_weight.dtype, device=class_weight.device
            )
            valid_mask = (label_flat != ignore_index) & (label_flat >= 0) & (
                label_flat < len(class_weight)
            )
            if valid_mask.any():
                valid_indices = label_flat[valid_mask]
                label_weights[valid_mask] = class_weight[valid_indices]

            label_weights = label_weights.view(label.shape)

            if avg_non_ignore:
                label_weights[label == ignore_index] = 0
            avg_factor = label_weights.sum()

    if weight is not None:
        weight = weight.float()

    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor
    )
    return loss


@MODELS.register_module()
class FixedCrossEntropyLoss(CrossEntropyLoss):
    """Same as MMSeg ``CrossEntropyLoss`` but uses :func:`fixed_cross_entropy` for softmax CE.

    Fixes average reduction when both ``class_weight`` and ``ignore_index`` are used.
    Tested with mmsegmentation 1.2.x.
    """

    def __init__(self, **kwargs):
        import mmseg as mmseg_mod

        ver = getattr(mmseg_mod, "__version__", "0.0.0")
        if not str(ver).startswith("1.2."):
            raise RuntimeError(
                f"FixedCrossEntropyLoss is validated for mmsegmentation 1.2.x; got {ver}. "
                "Pin mmsegmentation or verify fixed_cross_entropy against your version."
            )
        super().__init__(**kwargs)
        if not self.use_sigmoid and not self.use_mask:
            self.cls_criterion = fixed_cross_entropy
