"""Confidence-threshold filtering for segmentation masks."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from mowing_terrain_seg.utils.logger import LOGGER


def apply_confidence_threshold(
    mask: np.ndarray,
    confidence_scores: Optional[np.ndarray],
    conf_thresholds: Optional[Union[float, List[float]]],
    num_classes: Optional[int],
    raw_output,
) -> np.ndarray:
    if conf_thresholds is None:
        return mask

    if confidence_scores is None:
        if isinstance(raw_output, np.ndarray):
            output_shape = raw_output.shape
            if len(output_shape) >= 2 and (
                len(output_shape) == 2
                or (len(output_shape) == 3 and output_shape[0] == 1)
                or (len(output_shape) == 4 and output_shape[1] == 1)
            ):
                LOGGER.warning(
                    f"Confidence thresholds provided but ONNX model outputs predictions "
                    f"(shape: {output_shape}), not logits. Thresholding requires logits. "
                    f"Please export ONNX model with logits output (before ArgMax) to enable thresholding."
                )
            else:
                LOGGER.warning(
                    f"Confidence thresholds provided but could not extract logits from output "
                    f"(shape: {output_shape}). Thresholding will be skipped."
                )
        else:
            LOGGER.warning(
                "Confidence thresholds provided but no seg_logits found in model output. "
                "Thresholding will be skipped."
            )
        return mask

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    if num_classes is None or num_classes <= 0:
        return mask
    if isinstance(conf_thresholds, (int, float)):
        thresholds = [float(conf_thresholds)] * num_classes
    else:
        thresholds = list(conf_thresholds)

    filtered_mask = mask.copy()
    for class_id, threshold in enumerate(thresholds):
        if num_classes is not None and class_id >= num_classes:
            continue
        class_m = (mask == class_id)
        low = (confidence_scores < threshold) & class_m
        filtered_mask[low] = 255
    return filtered_mask
