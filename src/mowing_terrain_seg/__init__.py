# Copyright (c) OpenMMLab. All rights reserved.
"""
Mowing terrain segmentation package.
"""

# Import all modules to register them with MMSegmentation
from . import datasets
from . import models
from . import visualization
from . import engine
from . import evaluation

# Re-export all modules for convenience
from .datasets import *
from .models import *
from .visualization import *

__all__ = [
    'datasets',
    'models',
    'visualization',
    'engine',
    'evaluation',
]
