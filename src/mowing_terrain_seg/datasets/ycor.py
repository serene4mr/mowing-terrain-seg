import os
import numpy as np
import torch  # Move import to top level for optimization
from PIL import Image
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS

@DATASETS.register_module()
class YCORDataset(BaseSegDataset):
    """YCOR dataset for off-road navigation segmentation.
    
    In segmentation map annotation for YCOR, 0 stands for background, which
    is not included in 8 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    
    Classes:
        0: Background (ignored with reduce_zero_label=True)
        1: Smooth trail  
        2: Traversable grass
        3: Rough trail
        4: Puddle
        5: Obstacle
        6: Non-traversable vegetation
        7: High vegetation
        8: Sky
    """
    
    METAINFO = dict(
        classes=('smooth_trail', 'traversable_grass', 'rough_trail',
                'puddle', 'obstacle', 'non_traversable_vegetation', 'high_vegetation', 'sky'),
        palette=[[178, 176, 153], [128, 255, 0], [156, 76, 30],
                [255, 0, 128], [255, 0, 0], [0, 160, 0], [40, 80, 0], [1, 88, 255]]
    )
    
    def __init__(self, img_suffix='.jpg', seg_map_suffix='.png', reduce_zero_label=True, **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        self.reduce_zero_label = reduce_zero_label
    
    def load_data_list(self):
        """Load YCOR data structure: data/ycor/{train,valid}/iid*/{rgb.jpg, labels.png}
        
        Returns:
            list: List of data samples, each containing:
                - img_path: Path to RGB image
                - seg_map_path: Path to segmentation mask
                - split: 'train' or 'valid'
                - sample_id: Sample identifier (e.g., 'iid000000')
        """
        data_list = []
        data_root = self.data_root
        
        if not os.path.exists(data_root):
            raise FileNotFoundError(f"Data root directory not found: {data_root}")
        
        # Determine which split to load based on data_prefix
        if hasattr(self, 'data_prefix') and 'img_path' in self.data_prefix:
            # Extract just the split name from the full path
            img_path = self.data_prefix['img_path']
            if img_path.startswith(data_root):
                # Remove data_root prefix to get just the split name
                split_name = img_path[len(data_root):].lstrip('/')
            else:
                split_name = img_path
            splits_to_load = [split_name]
        else:
            # Default: load both splits (for backward compatibility)
            splits_to_load = ['train', 'valid']
        
        # Load from specified split(s)
        for split in splits_to_load:
            split_path = os.path.join(data_root, split)
            if os.path.exists(split_path):
                # Get all sample directories that start with 'iid'
                sample_dirs = [d for d in os.listdir(split_path) 
                             if d.startswith('iid') and os.path.isdir(os.path.join(split_path, d))]
                sample_dirs.sort()  # Sort for consistent ordering
                
                for sample_dir in sample_dirs:
                    sample_path = os.path.join(split_path, sample_dir)
                    rgb_file = os.path.join(sample_path, 'rgb.jpg')
                    labels_file = os.path.join(sample_path, 'labels.png')
                    
                    # Check if both required files exist
                    if os.path.exists(rgb_file) and os.path.exists(labels_file):
                        data_list.append({
                            'img_path': rgb_file,
                            'seg_map_path': labels_file,
                            'seg_fields': [],  # Required by MMSegmentation
                            'sample_idx': len(data_list),  # Required by MMSegmentation
                            'reduce_zero_label': self.reduce_zero_label,  # Required for LoadAnnotations transform
                            'split': split,  # Custom field for your dataset
                            'sample_id': sample_dir  # Custom field for your dataset
                        })
                    else:
                        # Log missing files for debugging
                        missing = []
                        if not os.path.exists(rgb_file):
                            missing.append('rgb.jpg')
                        if not os.path.exists(labels_file):
                            missing.append('labels.png')
                        print(f"Warning: Missing files in {sample_path}: {missing}")
        
        if not data_list:
            raise ValueError(f"No valid data samples found in {data_root}")
        
        print(f"Loaded {len(data_list)} samples from YCOR dataset")
        return data_list
    
    def _get_default_metainfo(self):
        """Get default metainfo for YCOR dataset"""
        return self.METAINFO


@DATASETS.register_module()
class YCORLawnMowing3ClassDataset(YCORDataset):
    """YCOR dataset grouped for lawn mowing application.
    
    Groups the original 8 YCOR classes into 3 categories:
    - Cuttable: Areas that can be mowed (grass, vegetation)
    - Traversable: Areas safe to drive/walk on (trails, paths)
    - Non-traversable: Obstacles, barriers, and unknown areas (obstacles, high vegetation, sky, background)
    
    Original class mapping:
        0: background -> Non-traversable (safety first - unknown areas)
        1: smooth_trail -> Traversable
        2: traversable_grass -> Cuttable
        3: rough_trail -> Traversable
        4: puddle -> Non-traversable (water hazard)
        5: obstacle -> Non-traversable
        6: non_traversable_vegetation -> Cuttable (can be mowed)
        7: high_vegetation -> Non-traversable (too high to mow)
        8: sky -> Non-traversable
    """
    
    METAINFO = dict(
        classes=('Cuttable', 'Traversable', 'Non-traversable'),
        palette=[[0, 255, 0], [178, 176, 153], [255, 0, 0]]  # Green, Gray, Red
    )
    
    def __init__(self, **kwargs):
        # Set reduce_zero_label=False since we're handling background mapping ourselves
        kwargs['reduce_zero_label'] = False
        super().__init__(**kwargs)        
        
        # Define label mapping: original_label -> new_label
        # Background (0) is mapped to Non-traversable (2) for safety
        self.label_map = {
            0: 2,  # background -> Non-traversable (safety first)
            1: 1,  # smooth_trail -> Traversable
            2: 0,  # traversable_grass -> Cuttable
            3: 1,  # rough_trail -> Traversable
            4: 2,  # puddle -> Non-traversable (water hazard)
            5: 2,  # obstacle -> Non-traversable
            6: 0,  # non_traversable_vegetation -> Cuttable (can be mowed)
            7: 2,  # high_vegetation -> Non-traversable (too high)
            8: 2,  # sky -> Non-traversable
        }
        
        # Pre-compute lookup table for faster remapping
        self._create_lookup_table()
    
    def _create_lookup_table(self):
        """Create a lookup table for O(1) label remapping."""
        # Create a lookup table for all possible label values (0-255)
        # Use int64 to match PyTorch's expected dtype for class indices
        self.lookup_table = np.zeros(256, dtype=np.int64)
        for old_label, new_label in self.label_map.items():
            self.lookup_table[old_label] = new_label
    
    def __getitem__(self, idx):
        """Get data sample with optimized label remapping."""
        results = super().__getitem__(idx)
        
        # Apply label mapping to segmentation mask
        if 'data_samples' in results:
            data_sample = results['data_samples']
            if hasattr(data_sample, 'gt_sem_seg'):
                gt_seg = data_sample.gt_sem_seg.data
                
                # Optimized remapping using lookup table
                if isinstance(gt_seg, torch.Tensor):
                    # Use tensor operations for better performance
                    gt_seg = self._remap_tensor(gt_seg)
                else:
                    # Fallback for numpy arrays
                    gt_seg = self._remap_numpy(gt_seg)
                
                data_sample.gt_sem_seg.data = gt_seg
        
        return results
    
    def _remap_tensor(self, gt_seg):
        """Optimized tensor remapping using vectorized operations."""
        # Convert to numpy for lookup, then back to tensor
        gt_seg_np = gt_seg.cpu().numpy()
        gt_seg_remapped = self.lookup_table[gt_seg_np]
        return torch.from_numpy(gt_seg_remapped).to(gt_seg.device)
    
    def _remap_numpy(self, gt_seg):
        """Optimized numpy remapping using lookup table."""
        return self.lookup_table[gt_seg]
