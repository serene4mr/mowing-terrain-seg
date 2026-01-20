# Copyright (c) OpenMMLab. All rights reserved.
"""
Custom datasets for visual segmentation benchmark.
"""

from .ycor import YCORDataset, YCORLawnMowing3ClassDataset

# Add more custom datasets here as you create them:
# from .custom_dataset2 import CustomDataset2
# from .custom_dataset3 import CustomDataset3

__all__ = [
    # Add new dataset names here
    'YCORDataset',
    'YCORLawnMowing3ClassDataset',
]
