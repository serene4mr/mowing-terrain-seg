# Analysis Tools

This directory contains tools for analyzing and visualizing datasets and model results.

## browse_dataset.py

A tool for browsing and visualizing dataset samples with their ground truth segmentation masks.

### Usage

#### Basic usage (save samples to directory):
```bash
python tools/analysis_tools/browse_dataset.py configs/deeplabv3/deeplabv3_r50-d8_4xb2-40k_ycor-1024x544.py --output-dir browse_output --not-show --max-samples 10
```

#### Interactive browsing (with display):
```bash
python tools/analysis_tools/browse_dataset.py configs/deeplabv3/deeplabv3_r50-d8_4xb2-40k_ycor-1024x544.py --show-interval 1 --max-samples 20
```

#### Browse validation dataset:
```bash
python tools/analysis_tools/browse_dataset.py configs/deeplabv3/deeplabv3_r50-d8_4xb2-40k_ycor-1024x544.py --cfg-options train_dataloader.dataset.data_prefix.img_path=valid train_dataloader.dataset.data_prefix.seg_map_path=valid --output-dir val_browse --not-show --max-samples 5
```

### Arguments

- `config`: Path to the training config file
- `--output-dir`: Directory to save visualization images (optional)
- `--not-show`: Don't display images interactively (default: False)
- `--show-interval`: Time interval between images in seconds (default: 2)
- `--max-samples`: Maximum number of samples to browse (default: 10)
- `--cfg-options`: Override config settings

### Output

The tool generates side-by-side images showing:
- **Left side**: Original RGB image
- **Right side**: Colored segmentation mask

Each class is assigned a unique color from the dataset palette:
- background: White
- smooth_trail: Light brown
- traversable_grass: Green
- rough_trail: Brown
- puddle: Pink
- obstacle: Red
- non_traversable_vegetation: Dark green
- high_vegetation: Darker green
- sky: Blue

### Interactive Controls

When using `--not-show` is False:
- Press any key to advance to next image
- Press 'q' to quit early

## analyze_dataset.py

A comprehensive tool for analyzing class distribution in training and validation datasets. This tool provides detailed statistics about pixel counts and percentages for each class, helping you understand dataset balance and identify potential issues.

### Usage

#### Console output only:
```bash
python tools/analysis_tools/analyze_dataset.py configs/deeplabv3/deeplabv3_r50-d8_4xb2-40k_ycor-1024x544.py
```

#### Save results to file:
```bash
python tools/analysis_tools/analyze_dataset.py configs/deeplabv3/deeplabv3_r50-d8_4xb2-40k_ycor-1024x544.py --output-dir analysis_results
```

When `--output-dir` is specified, the tool creates:
- `dataset_analysis.txt` - Complete analysis including training, validation, comparison, summary, and class weights
- `class_weights.txt` - Ready-to-use class weights in Python format for training

### Arguments

- `config`: Path to the training config file
- `--output-dir`: Directory to save analysis results (optional)
- `--cfg-options`: Override config settings (optional)

### Output

The tool provides comprehensive analysis in four sections:

#### 1. Training Dataset Analysis
- Total number of samples and pixels
- Class distribution with pixel counts and percentages
- Progress bar showing processing status

#### 2. Validation Dataset Analysis  
- Same analysis as training but for validation set
- Independent statistics for comparison

#### 3. Dataset Comparison
- Side-by-side comparison of class percentages
- Difference calculation between train and validation
- Helps identify distribution mismatches

#### 4. Summary Statistics
- Total samples and pixels across both datasets
- Overall dataset composition

#### 5. Class Weights Calculation
- **Inverse Frequency**: `weight = total_pixels / (num_classes * class_pixel_count)`
- **Square Root Inverse**: `weight = sqrt(total_pixels / class_pixel_count)`
- **Log Inverse**: `weight = log(total_pixels / class_pixel_count)`
- Separate calculations for training, validation, and combined datasets
- Ready-to-use Python arrays for implementing weighted loss functions

### Sample Output

```
============================================================
CLASS DISTRIBUTION ANALYSIS - TRAINING
============================================================
Class ID Class Name                Pixel Count     Percentage  
------------------------------------------------------------
0        background                11,559,624      3.37        %
1        smooth_trail              62,910,888      18.32       %
2        traversable_grass         42,272,451      12.31       %
3        rough_trail               57,356,919      16.70       %
4        puddle                    790,663         0.23        %
5        obstacle                  1,994,014       0.58        %
6        non_traversable_vegetation 17,104,536      4.98        %
7        high_vegetation           109,166,292     31.79       %
8        sky                       40,275,850      11.73       %
------------------------------------------------------------
TOTAL    All Classes               343,431,237     100.00%     

============================================================
DATASET COMPARISON
============================================================
Class                     Train %      Valid %      Difference  
------------------------------------------------------------
background                3.37         5.29         -1.92       
smooth_trail              18.32        13.92        4.39        
traversable_grass         12.31        15.56        -3.25       
...
```

### Use Cases

- **Class Imbalance Detection**: Identify classes with very low or high representation
- **Train/Validation Comparison**: Ensure similar distributions between splits
- **Data Augmentation Planning**: Determine which classes need more augmentation
- **Model Performance Analysis**: Understand why certain classes perform poorly
- **Dataset Quality Assessment**: Verify annotation consistency and completeness
- **Weighted Loss Implementation**: Get ready-to-use class weights for training with imbalanced datasets

### Using Class Weights in Training

The generated `class_weights.txt` file contains Python arrays that can be directly used in your training configuration:

```python
# Load the weights
exec(open('analysis_results/class_weights.txt').read())

# Use in your loss function (example for CrossEntropyLoss)
loss_decode = dict(
    type='CrossEntropyLoss', 
    use_sigmoid=False, 
    loss_weight=1.0,
    class_weight=inverse_frequency_weights  # or sqrt_inverse_weights, log_inverse_weights
)
```

**Weight Method Recommendations:**
- **Inverse Frequency**: Most aggressive rebalancing, use for severely imbalanced datasets
- **Square Root Inverse**: Moderate rebalancing, good general-purpose choice
- **Log Inverse**: Gentle rebalancing, preserves more of the original distribution
