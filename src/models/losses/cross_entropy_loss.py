# src/models/losses/cross_entropy_loss.py
import torch
import torch.nn.functional as F
from mmseg.models.losses import cross_entropy_loss
from mmseg.models.losses.utils import weight_reduce_loss


def fixed_cross_entropy(pred,
                        label,
                        weight=None,
                        class_weight=None,
                        reduction='mean',
                        avg_factor=None,
                        ignore_index=-100,
                        avg_non_ignore=False):
    """Fixed cross_entropy that properly handles ignore_index with class_weight."""
    
    # Use original F.cross_entropy (this works fine)
    loss = F.cross_entropy(
        pred, label, weight=class_weight, reduction='none', ignore_index=ignore_index)

    # Apply weights and do the reduction
    if (avg_factor is None) and reduction == 'mean':
        if class_weight is None:
            if avg_non_ignore:
                avg_factor = label.numel() - (label == ignore_index).sum().item()
            else:
                avg_factor = label.numel()
        else:
            # ✅ FIXED CODE - Replace the buggy lines 71-73
            label_flat = label.view(-1)
            label_weights = torch.zeros_like(label_flat, dtype=class_weight.dtype, device=class_weight.device)
            
            # Only process valid class indices
            valid_mask = (label_flat != ignore_index) & (label_flat >= 0) & (label_flat < len(class_weight))
            if valid_mask.any():
                valid_indices = label_flat[valid_mask]
                label_weights[valid_mask] = class_weight[valid_indices]
            
            label_weights = label_weights.view(label.shape)

            if avg_non_ignore:
                label_weights[label == ignore_index] = 0
            avg_factor = label_weights.sum()

    if weight is not None:
        weight = weight.float()
    
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss

cross_entropy_loss.cross_entropy = fixed_cross_entropy