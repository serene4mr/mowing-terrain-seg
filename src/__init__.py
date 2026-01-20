# Copyright (c) OpenMMLab. All rights reserved.
"""
Custom datasets package for visual segmentation benchmark.
"""

# Import all modules to register them with MMSegmentation
from . import datasets
from . import models
from . import visualization

# Re-export all modules for convenience
from .datasets import *
from .models import *
from .visualization import *

__all__ = [
    'datasets',
    'models',
    'visualization',
]
