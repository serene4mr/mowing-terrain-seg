# Yamaha Segmentation Dataset Analysis

## Overview
The Yamaha segmentation dataset is a collection of off-road terrain images with corresponding semantic segmentation masks. This dataset is designed for terrain classification and segmentation tasks in off-road environments.

## Dataset Structure
```
yamaha_seg/
└── yamaha_v0/
    ├── train/          # Training set
    │   ├── iid000000/
    │   │   ├── rgb.jpg     # RGB image
    │   │   └── labels.png  # Segmentation mask
    │   ├── iid000001/
    │   └── ... (931 samples)
    └── valid/          # Validation set
        ├── iid000839/
        │   ├── rgb.jpg     # RGB image
        │   └── labels.png  # Segmentation mask
        └── ... (145 samples)
```

## Dataset Statistics

| Split | Samples | Size | Percentage |
|-------|---------|------|------------|
| **Train** | 931 | 143 MB | 86.5% |
| **Valid** | 145 | 19 MB | 13.5% |
| **Total** | 1,076 | 161 MB | 100% |

## File Information

### Images (rgb.jpg)
- **Format**: JPEG
- **Total Count**: 1,076
- **Total Size**: 145.3 MB
- **Average Size**: 138.3 KB
- **Size Range**: 34 KB - 290 KB
- **Compression**: RGB color images

### Masks (labels.png)
- **Format**: PNG (indexed color)
- **Total Count**: 1,076
- **Total Size**: 6.5 MB
- **Average Size**: 6.2 KB
- **Size Range**: 2.7 KB - 18.6 KB
- **Compression**: Lossless segmentation labels
- **Data Type**: numpy.uint8
- **Value Range**: 0-8 (9 classes)
- **Palette**: 256-color indexed palette (only 9 colors used)

## Naming Convention
- **Sample IDs**: `iidXXXXXXXX` (8-digit identifier)
- **Training Range**: `iid000000` to `iid001343`
- **Validation Range**: `iid000839` to `iid001052`
- **File Structure**: Each sample folder contains exactly one `rgb.jpg` and one `labels.png`

## Dataset Characteristics
- **Domain**: Off-road terrain
- **Purpose**: Semantic segmentation
- **Image-Mask Pairing**: 1:1 correspondence
- **Consistency**: All samples follow identical structure
- **Quality**: High-quality RGB images with precise segmentation masks
- **Mask Format**: Indexed color PNG with consistent palette
- **Value Consistency**: Non-consecutive class indices (missing class 4 in some samples)
- **Variable Ranges**: Some masks have range 0-7, others 0-8

## Source
- **Original Source**: Yamaha Research
- **Download Link**: [CMU Box](https://cmu.app.box.com/s/3fngoljhcwhqf2z5cbepufh331qtesxt)
- **Version**: yamaha_v0
- **Date**: January 2021

## Class Information
- **Total Classes**: 9 (values 0-8)
- **Class Values**: 0, 1, 2, 3, 4, 5, 6, 7, 8
- **Data Type**: numpy.uint8
- **Color Format**: Indexed color PNG masks (palette-based)

### Color Palette
| Class | RGB Values | Hex Color | Color Name | Class Description |
|-------|------------|-----------|------------|---------------------------|
| 0 | (255,255,255) | #ffffff | White | Background/unwanted classes |
| 1 | (178,176,153) | #b2b099 | Light Gray | Smooth trail |
| 2 | (128,255,0) | #80ff00 | Lime Green | Traversable grass |
| 3 | (156,76,30) | #9c4c1e | Brown | Rough trail |
| 4 | (255,0,128) | #ff0080 | Magenta | Puddle |
| 5 | (255,0,0) | #ff0000 | Red | Obstacle |
| 6 | (0,160,0) | #00a000 | Green | Non-traversable low vegetation |
| 7 | (40,80,0) | #285000 | Dark Green | High vegetation |
| 8 | (1,88,255) | #0158ff | Blue | Sky |

**Note**: Class descriptions are based on color analysis and typical off-road terrain segmentation patterns.

## Class Distribution Analysis

### Per-Image Class Occurrence
| Class | Train Images | Train % | Val Images | Val % | Difference |
|-------|-------------|---------|------------|-------|------------|
| 0     | 931         | 100.0%  | 145        | 100.0%| 0.0%       |
| 1     | 549         | 59.0%   | 85         | 58.6% | -0.3%      |
| 2     | 642         | 69.0%   | 103        | 71.0% | +2.1%      |
| 3     | 520         | 55.9%   | 83         | 57.2% | +1.4%      |
| 4     | 42          | 4.5%    | 2          | 1.4%  | -3.1%      |
| 5     | 164         | 17.6%   | 42         | 29.0% | +11.4%     |
| 6     | 355         | 38.1%   | 69         | 47.6% | +9.5%      |
| 7     | 923         | 99.1%   | 144        | 99.3% | +0.2%      |
| 8     | 844         | 90.7%   | 139        | 95.9% | +5.2%      |

### Pixel-Level Distribution
| Class | Train Pixels | Train % | Val Pixels | Val % | Difference |
|-------|-------------|---------|------------|-------|------------|
| 0     | 14,837,133  | 2.9%    | 4,269,797  | 5.3%  | +2.4%      |
| 1     | 78,010,390  | 15.0%   | 11,247,164 | 13.9% | -1.1%      |
| 2     | 69,291,002  | 13.4%   | 12,565,073 | 15.6% | +2.2%      |
| 3     | 78,813,041  | 15.2%   | 12,481,369 | 15.5% | +0.3%      |
| 4     | 954,359     | 0.2%    | 110,494    | 0.1%  | -0.1%      |
| 5     | 4,489,219   | 0.9%    | 1,526,245  | 1.9%  | +1.0%      |
| 6     | 32,365,180  | 6.2%    | 5,085,590  | 6.3%  | +0.1%      |
| 7     | 187,016,896 | 36.1%   | 25,155,266 | 31.1% | **-4.9%**  |
| 8     | 52,841,916  | 10.2%   | 8,332,122  | 10.3% | +0.1%      |

### Key Findings
- **Class Imbalance**: Significant imbalance exists, especially for classes 4 and 5
- **Distribution Bias**: Classes 5, 6, and 8 show notable differences between train/val sets
- **Most Common**: Class 7 dominates with 36.1% of pixels in training
- **Rarest Classes**: Class 4 appears in only 4.5% of training images and 0.2% of pixels
- **Balanced Classes**: Classes 0, 1, 3, and 7 show good train/val balance
- **Total Pixels**: 518.6M training pixels, 80.8M validation pixels

## Usage Notes
- Images and masks are perfectly paired
- Each sample represents a unique terrain scene
- Masks contain pixel-level segmentation labels
- Dataset is ready for semantic segmentation tasks
- **Class Imbalance Warning**: Significant class imbalance may require special handling
- **Distribution Bias**: Some classes show different distributions between train/val sets

