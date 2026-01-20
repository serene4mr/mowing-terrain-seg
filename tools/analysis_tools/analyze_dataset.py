
import argparse
import os
import os.path as osp
import sys
import numpy as np
from collections import Counter
import math

# Add src directory to Python path for custom datasets
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import all custom datasets to register them
import src

from mmengine.config import Config, DictAction
from mmengine.utils import ProgressBar

from mmseg.registry import DATASETS
from mmseg.utils import register_all_modules


def calculate_class_weights(class_counts, total_pixels, num_classes):
    """Calculate class weights using different methods"""
    weights = {}
    
    # Get pixel counts for each class (including classes with 0 pixels)
    pixel_counts = [class_counts.get(i, 0) for i in range(num_classes)]
    
    # Method 1: Inverse Frequency
    # weight = total_pixels / (num_classes * class_pixel_count)
    inverse_freq_weights = []
    for count in pixel_counts:
        if count > 0:
            weight = total_pixels / (num_classes * count)
        else:
            weight = 0.0  # Handle classes with no pixels
        inverse_freq_weights.append(weight)
    
    # Method 2: Square Root Inverse
    # weight = sqrt(total_pixels / class_pixel_count)
    sqrt_inverse_weights = []
    for count in pixel_counts:
        if count > 0:
            weight = math.sqrt(total_pixels / count)
        else:
            weight = 0.0
        sqrt_inverse_weights.append(weight)
    
    # Method 3: Log Inverse
    # weight = log(total_pixels / class_pixel_count)
    log_inverse_weights = []
    for count in pixel_counts:
        if count > 0:
            weight = math.log(total_pixels / count)
        else:
            weight = 0.0
        log_inverse_weights.append(weight)
    
    weights['inverse_frequency'] = inverse_freq_weights
    weights['sqrt_inverse'] = sqrt_inverse_weights
    weights['log_inverse'] = log_inverse_weights
    
    return weights


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Dataset')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='Directory to save analysis results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules in mmseg into the registries
    register_all_modules()
    
    # Capture all output if output_dir is specified
    output_lines = []
    
    def print_and_save(text):
        print(text)
        if args.output_dir:
            output_lines.append(text)
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print_and_save(f"Output directory: {args.output_dir}")
    
    print_and_save("Loading datasets...")
    train_dataset = DATASETS.build(cfg.train_dataloader.dataset)
    valid_dataset = DATASETS.build(cfg.val_dataloader.dataset)
    
    # Analyze training dataset
    print_and_save(f"\n{'='*60}")
    print_and_save(f"ANALYZING TRAINING DATASET")
    print_and_save(f"{'='*60}")
    
    # Get dataset metadata
    metainfo = train_dataset.metainfo
    classes = metainfo.get('classes', [])
    
    print_and_save(f"Dataset: TRAINING")
    print_and_save(f"Total samples: {len(train_dataset)}")
    print_and_save(f"Number of classes: {len(classes)}")
    print_and_save(f"Classes: {classes}")
    
    # Count pixels for each class
    train_class_pixel_counts = Counter()
    train_total_pixels = 0
    
    print_and_save(f"\nProcessing {len(train_dataset)} samples...")
    progress_bar = ProgressBar(len(train_dataset))
    
    for i, item in enumerate(train_dataset):
        # Get segmentation mask
        seg_data = item['data_samples'].gt_sem_seg.data.squeeze()
        if hasattr(seg_data, 'numpy'):
            seg_map = seg_data.numpy()
        else:
            seg_map = seg_data
        
        # Count pixels for each class
        unique, counts = np.unique(seg_map, return_counts=True)
        for class_id, count in zip(unique, counts):
            train_class_pixel_counts[class_id] += count
            train_total_pixels += count
        
        progress_bar.update()
    
    # Calculate percentages
    print_and_save(f"\n{'='*60}")
    print_and_save(f"CLASS DISTRIBUTION ANALYSIS - TRAINING")
    print_and_save(f"{'='*60}")
    print_and_save(f"{'Class ID':<8} {'Class Name':<25} {'Pixel Count':<15} {'Percentage':<12}")
    print_and_save(f"{'-'*60}")
    
    for class_id in sorted(train_class_pixel_counts.keys()):
        class_name = classes[class_id] if class_id < len(classes) else f"Unknown_{class_id}"
        pixel_count = train_class_pixel_counts[class_id]
        percentage = (pixel_count / train_total_pixels) * 100 if train_total_pixels > 0 else 0
        
        print_and_save(f"{class_id:<8} {class_name:<25} {pixel_count:<15,} {percentage:<12.2f}%")
    
    print_and_save(f"{'-'*60}")
    print_and_save(f"{'TOTAL':<8} {'All Classes':<25} {train_total_pixels:<15,} {'100.00%':<12}")
    
    # Analyze validation dataset
    print_and_save(f"\n{'='*60}")
    print_and_save(f"ANALYZING VALIDATION DATASET")
    print_and_save(f"{'='*60}")
    
    print_and_save(f"Dataset: VALIDATION")
    print_and_save(f"Total samples: {len(valid_dataset)}")
    print_and_save(f"Number of classes: {len(classes)}")
    print_and_save(f"Classes: {classes}")
    
    # Count pixels for each class
    valid_class_pixel_counts = Counter()
    valid_total_pixels = 0
    
    print_and_save(f"\nProcessing {len(valid_dataset)} samples...")
    progress_bar = ProgressBar(len(valid_dataset))
    
    for i, item in enumerate(valid_dataset):
        # Get segmentation mask
        seg_data = item['data_samples'].gt_sem_seg.data.squeeze()
        if hasattr(seg_data, 'numpy'):
            seg_map = seg_data.numpy()
        else:
            seg_map = seg_data
        
        # Count pixels for each class
        unique, counts = np.unique(seg_map, return_counts=True)
        for class_id, count in zip(unique, counts):
            valid_class_pixel_counts[class_id] += count
            valid_total_pixels += count
        
        progress_bar.update()
    
    # Calculate percentages
    print_and_save(f"\n{'='*60}")
    print_and_save(f"CLASS DISTRIBUTION ANALYSIS - VALIDATION")
    print_and_save(f"{'='*60}")
    print_and_save(f"{'Class ID':<8} {'Class Name':<25} {'Pixel Count':<15} {'Percentage':<12}")
    print_and_save(f"{'-'*60}")
    
    for class_id in sorted(valid_class_pixel_counts.keys()):
        class_name = classes[class_id] if class_id < len(classes) else f"Unknown_{class_id}"
        pixel_count = valid_class_pixel_counts[class_id]
        percentage = (pixel_count / valid_total_pixels) * 100 if valid_total_pixels > 0 else 0
        
        print_and_save(f"{class_id:<8} {class_name:<25} {pixel_count:<15,} {percentage:<12.2f}%")
    
    print_and_save(f"{'-'*60}")
    print_and_save(f"{'TOTAL':<8} {'All Classes':<25} {valid_total_pixels:<15,} {'100.00%':<12}")
    
    # Compare datasets
    print_and_save(f"\n{'='*60}")
    print_and_save("DATASET COMPARISON")
    print_and_save(f"{'='*60}")
    print_and_save(f"{'Class':<25} {'Train %':<12} {'Valid %':<12} {'Difference':<12}")
    print_and_save(f"{'-'*60}")
    
    all_classes = set(train_class_pixel_counts.keys()) | set(valid_class_pixel_counts.keys())
    for class_id in sorted(all_classes):
        class_name = classes[class_id] if class_id < len(classes) else f"Unknown_{class_id}"
        
        train_pct = (train_class_pixel_counts[class_id] / train_total_pixels) * 100 if train_total_pixels > 0 else 0
        valid_pct = (valid_class_pixel_counts[class_id] / valid_total_pixels) * 100 if valid_total_pixels > 0 else 0
        diff = train_pct - valid_pct
        
        print_and_save(f"{class_name:<25} {train_pct:<12.2f} {valid_pct:<12.2f} {diff:<12.2f}")
    
    # Summary
    print_and_save(f"\n{'='*60}")
    print_and_save("SUMMARY")
    print_and_save(f"{'='*60}")
    print_and_save(f"Training samples: {len(train_dataset):,}")
    print_and_save(f"Validation samples: {len(valid_dataset):,}")
    print_and_save(f"Training pixels: {train_total_pixels:,}")
    print_and_save(f"Validation pixels: {valid_total_pixels:,}")
    print_and_save(f"Total samples: {len(train_dataset) + len(valid_dataset):,}")
    print_and_save(f"Total pixels: {train_total_pixels + valid_total_pixels:,}")
    
    # Calculate class weights
    print_and_save(f"\n{'='*60}")
    print_and_save("CLASS WEIGHTS CALCULATION FOR WEIGHTED LOSS TO HANDLE CLASS IMBALANCE")
    print_and_save(f"{'='*60}")
    
    # Calculate weights for training dataset
    train_weights = calculate_class_weights(train_class_pixel_counts, train_total_pixels, len(classes))
    
    print_and_save("Training Dataset Class Weights:")
    print_and_save(f"{'Class ID':<8} {'Class Name':<25} {'Inverse Freq':<15} {'Sqrt Inverse':<15} {'Log Inverse':<15}")
    print_and_save(f"{'-'*80}")
    
    for i, class_name in enumerate(classes):
        inverse_freq = train_weights['inverse_frequency'][i]
        sqrt_inv = train_weights['sqrt_inverse'][i]
        log_inv = train_weights['log_inverse'][i]
        
        print_and_save(f"{i:<8} {class_name:<25} {inverse_freq:<15.4f} {sqrt_inv:<15.4f} {log_inv:<15.4f}")
    
    # Calculate weights for validation dataset
    valid_weights = calculate_class_weights(valid_class_pixel_counts, valid_total_pixels, len(classes))
    
    print_and_save(f"\nValidation Dataset Class Weights:")
    print_and_save(f"{'Class ID':<8} {'Class Name':<25} {'Inverse Freq':<15} {'Sqrt Inverse':<15} {'Log Inverse':<15}")
    print_and_save(f"{'-'*80}")
    
    for i, class_name in enumerate(classes):
        inverse_freq = valid_weights['inverse_frequency'][i]
        sqrt_inv = valid_weights['sqrt_inverse'][i]
        log_inv = valid_weights['log_inverse'][i]
        
        print_and_save(f"{i:<8} {class_name:<25} {inverse_freq:<15.4f} {sqrt_inv:<15.4f} {log_inv:<15.4f}")
    
    # Add class weights to the output
    print_and_save(f"\n{'='*60}")
    print_and_save("CLASS WEIGHTS FOR TRAINING")
    print_and_save(f"{'='*60}")
    
    print_and_save("# ===== TRAINING DATASET WEIGHTS =====\n")
    print_and_save("# Inverse Frequency Weights (Training)")
    print_and_save("inverse_frequency_weights_train = [")
    for i, weight in enumerate(train_weights['inverse_frequency']):
        print_and_save(f"{weight:.6f}" + ("," if i < len(train_weights['inverse_frequency']) - 1 else ""))
    print_and_save("]\n")
    
    print_and_save("# Square Root Inverse Weights (Training)")
    print_and_save("sqrt_inverse_weights_train = [")
    for i, weight in enumerate(train_weights['sqrt_inverse']):
        print_and_save(f"{weight:.6f}" + ("," if i < len(train_weights['sqrt_inverse']) - 1 else ""))
    print_and_save("]\n")
    
    print_and_save("# Log Inverse Weights (Training)")
    print_and_save("log_inverse_weights_train = [")
    for i, weight in enumerate(train_weights['log_inverse']):
        print_and_save(f"{weight:.6f}" + ("," if i < len(train_weights['log_inverse']) - 1 else ""))
    print_and_save("]\n")
    
    print_and_save("# ===== VALIDATION DATASET WEIGHTS =====\n")
    print_and_save("# Inverse Frequency Weights (Validation)")
    print_and_save("inverse_frequency_weights_valid = [")
    for i, weight in enumerate(valid_weights['inverse_frequency']):
        print_and_save(f"{weight:.6f}" + ("," if i < len(valid_weights['inverse_frequency']) - 1 else ""))
    print_and_save("]\n")
    
    print_and_save("# Square Root Inverse Weights (Validation)")
    print_and_save("sqrt_inverse_weights_valid = [")
    for i, weight in enumerate(valid_weights['sqrt_inverse']):
        print_and_save(f"{weight:.6f}" + ("," if i < len(valid_weights['sqrt_inverse']) - 1 else ""))
    print_and_save("]\n")
    
    print_and_save("# Log Inverse Weights (Validation)")
    print_and_save("log_inverse_weights_valid = [")
    for i, weight in enumerate(valid_weights['log_inverse']):
        print_and_save(f"{weight:.6f}" + ("," if i < len(valid_weights['log_inverse']) - 1 else ""))
    print_and_save("]\n")
    
    print_and_save("# ===== CLASS NAMES =====\n")
    print_and_save("class_names = [")
    for i, class_name in enumerate(classes):
        print_and_save(f"'{class_name}'" + ("," if i < len(classes) - 1 else ""))
    print_and_save("]")
    
    # Save to single file if output_dir is specified
    if args.output_dir:
        output_file = os.path.join(args.output_dir, "dataset_analysis.txt")
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"\nAnalysis saved to: {output_file}")


if __name__ == '__main__':
    main()