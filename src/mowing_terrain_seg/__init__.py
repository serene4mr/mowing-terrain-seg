# Copyright (c) OpenMMLab. All rights reserved.
"""
Mowing terrain segmentation package.
"""

# Import all modules to register them with MMSegmentation
from . import datasets
from . import models
from . import visualization

# Re-export all modules for convenience
from .datasets import *
from .models import *
from .visualization import *


def register_all() -> None:
    """Call once from entrypoints: register mmseg + default scope, then this package is loaded."""
    from mmseg.utils import register_all_modules

    register_all_modules()


__all__ = [
    "datasets",
    "models",
    "visualization",
    "register_all",
]
