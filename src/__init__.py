# Copyright (c) OpenMMLab. All rights reserved.
"""
Custom datasets package for visual segmentation benchmark.
"""

# Import all modules to register them with MMSegmentation
from .mowing_terrain_seg import datasets
from .mowing_terrain_seg import models
from .mowing_terrain_seg import visualization

# Re-export all modules for convenience
from .mowing_terrain_seg.datasets import *
from .mowing_terrain_seg.models import *
from .mowing_terrain_seg.visualization import *

__all__ = [
    'datasets',
    'models',
    'visualization',
]
