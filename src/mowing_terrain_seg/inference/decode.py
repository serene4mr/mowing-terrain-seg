"""Decode raw model outputs (Torch DataSample or ONNX ndarray) to masks and confidence."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def extract_mask_from_numpy(raw_output: np.ndarray) -> np.ndarray:
    """Argmax / strip batch from ONNX ``ort_outputs[0]``."""
    output = raw_output
    if len(output.shape) == 4:
        output = output[0]

    if len(output.shape) == 3:
        if output.shape[0] == 1:
            mask = output[0].astype(np.uint8)
        else:
            mask = np.argmax(output, axis=0).astype(np.uint8)
    elif len(output.shape) == 2:
        mask = output.astype(np.uint8)
    else:
        raise ValueError(f"Unexpected ONNX output shape: {output.shape}")
    return mask


def extract_mask_from_data_sample(raw_output: Any) -> np.ndarray:
    """``pred_sem_seg`` from MMSeg ``SegDataSample``."""
    if hasattr(raw_output, "pred_sem_seg") and raw_output.pred_sem_seg is not None:
        mask = raw_output.pred_sem_seg.data.cpu().numpy()
        if len(mask.shape) == 3:
            mask = mask[0]
        return mask
    if hasattr(raw_output, "pred_instances"):
        return raw_output
    raise ValueError(
        f"No segmentation mask found in output. Available attributes: {dir(raw_output)}"
    )


def compute_confidence_numpy(raw_output: np.ndarray) -> Optional[np.ndarray]:
    """Max softmax probability per pixel from logits (C,H,W), or None if predictions only."""
    output = raw_output
    if len(output.shape) == 4:
        output = output[0]

    if len(output.shape) == 3:
        if output.shape[0] == 1:
            return None
        output_float = output.astype(np.float32)
        logits_tensor = torch.from_numpy(output_float)
        probs = F.softmax(logits_tensor, dim=0).numpy()
        return np.max(probs, axis=0)
    return None


def compute_confidence_data_sample(raw_output: Any) -> Optional[np.ndarray]:
    if not hasattr(raw_output, "seg_logits") or raw_output.seg_logits is None:
        return None
    logits = raw_output.seg_logits.data.cpu().numpy()
    logits_tensor = torch.from_numpy(logits)
    probs = F.softmax(logits_tensor, dim=0).numpy()
    return np.max(probs, axis=0)


def extract_mask_unified(
    raw_output: Union[np.ndarray, Any]
) -> np.ndarray:
    if isinstance(raw_output, np.ndarray):
        return extract_mask_from_numpy(raw_output)
    return extract_mask_from_data_sample(raw_output)


def compute_confidence_unified(
    raw_output: Union[np.ndarray, Any]
) -> Optional[np.ndarray]:
    if isinstance(raw_output, np.ndarray):
        return compute_confidence_numpy(raw_output)
    return compute_confidence_data_sample(raw_output)
