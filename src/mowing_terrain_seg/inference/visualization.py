"""Overlay and palette utilities for inference (OpenCV + optional PIL color names)."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import ImageColor


def get_auto_palette(num_classes: int) -> List[List[int]]:
    """HSV wheel palette (RGB triples) for ``num_classes``."""
    if num_classes is None or num_classes <= 0:
        return []
    hsv_colors = np.zeros((num_classes, 1, 3), dtype=np.uint8)
    hsv_colors[:, 0, 0] = np.linspace(30, 179, num_classes, endpoint=False)
    hsv_colors[:, 0, 1] = 200
    hsv_colors[:, 0, 2] = 255
    bgr_palette = cv2.cvtColor(hsv_colors, cv2.COLOR_HSV2BGR).reshape(-1, 3)
    rgb_palette = bgr_palette[:, [2, 1, 0]].tolist()
    return rgb_palette


def visualize_mask(
    img: np.ndarray,
    mask: np.ndarray,
    opacity: float = 0.7,
    palette: Optional[
        Union[
            List[List[int]],
            List[Tuple[int, int, int]],
            List[str],
        ]
    ] = None,
) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(
            mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    if palette is not None:
        max_class_id = int(mask.max())
        num_classes = len(palette)
        colormap_array = np.zeros(
            (max(max_class_id + 1, num_classes), 3), dtype=np.uint8
        )
        for class_id, color in enumerate(palette):
            if class_id < len(colormap_array):
                if isinstance(color, str):
                    rgb = ImageColor.getrgb(color)
                else:
                    rgb = tuple(color)
                if rgb == (0, 0, 0):
                    logging.warning(
                        f"Class ID {class_id} is assigned black (0, 0, 0) in the palette. "
                        "This conflicts with the reserved background/filtered color."
                    )
                colormap_array[class_id] = [rgb[2], rgb[1], rgb[0]]
        colored_mask = colormap_array[mask]
    else:
        colored_mask = np.zeros_like(img)
        valid_pixels = mask < 255
        colored_mask[valid_pixels] = [0, 255, 0]

    return cv2.addWeighted(img, 1 - opacity, colored_mask, opacity, 0)
