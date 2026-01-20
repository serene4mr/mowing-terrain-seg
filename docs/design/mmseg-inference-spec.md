# MMSegmentation Custom Inference Script - Development Specification

## Overview
Develop a comprehensive inference script for MMSegmentation framework that handles multiple input types (images, videos, directories) with configurable output options and robust error handling.

## Requirements

### Core Interface
```bash
python inference.py [REQUIRED_ARGS] [OPTIONAL_ARGS]
```

#### Required Arguments
- `--input`, `-i`: Path to input (image file, video file, or directory)
- `--config`, `-c`: Path to model config file (.py)
- `--checkpoint`, `-ckpt`: Path to model checkpoint file (.pth)
- `--output-dir`, `-o`: Directory to save all results

#### Optional Arguments
- `--no-save`: Disable saving any output files (default: False, saves overlays)
- `--save-masks`: Save raw prediction masks (default: False)
- `--show`: Display results on-screen during processing (default: False)
- `--wait-time`: Delay in ms between displayed frames when showing (default: 1)
- `--batch-size`: Number of images to process per batch (default: 1)
- `--device`: Compute device (default: 'cuda:0')
- `--opacity`: Overlay opacity for visualization (0.0-1.0, default: 0.7)
- `--overlay-fps`: Embed measured FPS text on result images/videos (default: False)
- `--img-extensions`: Image file extensions to process (default: ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'])
- `--vid-extensions`: Video file extensions to process (default: ['.mp4', '.avi', '.mov', '.mkv', '.wmv'])

## Architecture Design

### 1. Main Components

#### 1.1 ArgumentParser
- Use `argparse` with `RawDescriptionHelpFormatter`
- Include detailed help text with usage examples
- Validate required arguments presence
- Set appropriate default values

#### 1.2 FileProcessor Class
```python
class FileProcessor:
    def __init__(self, img_extensions, vid_extensions)
    def classify_input(self, input_path) -> dict
    def _classify_single_file(self, file_path) -> dict
    def _classify_directory(self, dir_path) -> dict
```

**Responsibilities:**
- Accept single file or directory path
- Recursively scan directories for supported files
- Return organized dictionary: `{'images': [paths...], 'videos': [paths...]}`
- Validate file extensions against allowed lists

#### 1.3 OutputManager Class
```python
class OutputManager:
    def __init__(self, output_dir, save_masks=False)
    def _create_directories(self)
    def get_output_paths(self, input_path, file_type) -> tuple
    def _generate_timestamp(self) -> str
```

**Responsibilities:**
- Create output directory structure
- Generate timestamped filenames preserving original stems
- Handle masks subdirectory creation when needed
- Return appropriate output paths for saving

#### 1.4 MMSegInference Class
```python
class MMSegInference:
    def __init__(self, config_path, checkpoint_path, device='cuda:0')
    def validate_model_config(self) -> bool
    def process_image(self, img_path, output_manager, **kwargs)
    def process_images_batch(self, img_paths, output_manager, **kwargs)
    def process_video(self, vid_path, output_manager, **kwargs)
    def _measure_fps(self, start_time, count) -> float
    def _overlay_fps_on_frame(self, frame, fps) -> np.ndarray
```

**Responsibilities:**
- Load and validate MMSegmentation model
- Handle single image, batch image, and video inference
- Measure and optionally overlay FPS on results
- Save visualization overlays and optional masks
- Integrate with on-screen display when requested

### 2. Processing Workflow

#### 2.1 Initialization
1. Parse command line arguments
2. Initialize logging with timestamps and INFO level
3. Create FileProcessor with extension filters
4. Create OutputManager with specified output directory
5. Validate and load MMSegmentation model

#### 2.2 File Processing
1. Classify input using FileProcessor
2. Log discovered files count
3. Process images (batch or single mode)
4. Process videos frame-by-frame
5. Handle exceptions with retry logic
6. Log completion statistics

#### 2.3 Error Handling
- Wrap I/O operations in try-except blocks
- Retry transient errors up to 2 times
- Log errors to console and optional log file
- Continue processing remaining files on individual failures
- Validate config-checkpoint compatibility before inference

## Technical Implementation Details

### 3. Dependencies
```python
import argparse
import logging
import time
import cv2
import mmcv
import numpy as np
from pathlib import Path
from tqdm import tqdm
from mmseg.apis import init_model, inference_model, MMSegInferencer
```

### 4. Key Functions

#### 4.1 FPS Measurement
```python
def measure_inference_fps(start_time, frame_count):
    elapsed = time.perf_counter() - start_time
    return frame_count / elapsed if elapsed > 0 else 0.0
```

#### 4.2 FPS Overlay
```python
def overlay_fps(frame, fps, position=(10, 30), 
                font=cv2.FONT_HERSHEY_SIMPLEX, 
                scale=1.0, color=(0, 255, 0), thickness=2):
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, position, font, scale, color, thickness)
    return frame
```

#### 4.3 Config Validation
- Parse model architecture from config file
- Compare with checkpoint metadata
- Raise clear error message on mismatch
- Log successful validation

#### 4.4 Progress Reporting
- Use `tqdm` for video frame processing progress bars
- Log batch processing statistics
- Include timing information in logs
- Report total files processed and any failures

### 5. Output File Naming Convention

#### Format: `<original_stem>_result_<timestamp>.<original_extension>`

Examples:
- Input: `forest_scene.jpg` → Output: `forest_scene_result_20250929_170500.jpg`
- Input: `drive_test.mp4` → Output: `drive_test_result_20250929_170500.mp4`

#### Mask Files (when --save-masks enabled):
- Images: `<original_stem>_mask_<timestamp>.png`
- Videos: `<original_stem>_frames_<timestamp>/frame_XXXXXX.png`

### 6. Directory Structure

```
output_dir/
├── image1_result_20250929_170500.jpg     # Visualization overlays
├── video1_result_20250929_170500.mp4     # Processed videos
└── masks/                                # Only if --save-masks
    ├── image1_mask_20250929_170500.png   # Raw image masks
    └── video1_frames_20250929_170500/    # Video frame masks
        ├── frame_000001.png
        ├── frame_000002.png
        └── ...
```

## Usage Examples

### Single Image
```bash
python inference.py \
    --input demo.jpg \
    --config configs/deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024.py \
    --checkpoint checkpoints/deeplabv3plus_r18.pth \
    --output-dir results/
```

### Video with Display and FPS Overlay
```bash
python inference.py \
    --input video.mp4 \
    --config configs/model.py \
    --checkpoint checkpoints/model.pth \
    --output-dir results/ \
    --show \
    --overlay-fps \
    --opacity 0.6
```

### Directory Processing with Masks
```bash
python inference.py \
    --input /path/to/mixed/data/ \
    --config configs/model.py \
    --checkpoint checkpoints/model.pth \
    --output-dir results/ \
    --save-masks \
    --batch-size 4
```

### Display Only (No Saving)
```bash
python inference.py \
    --input test_data/ \
    --config configs/model.py \
    --checkpoint checkpoints/model.pth \
    --output-dir /tmp/unused \
    --no-save \
    --show \
    --overlay-fps
```

## Quality Requirements

### Performance
- Efficient batch processing for multiple images
- Memory-conscious video frame processing
- GPU utilization optimization
- Progress reporting for long-running operations

### Robustness
- Graceful handling of corrupted input files
- Automatic retry on transient I/O errors
- Clear error messages for user-correctable issues
- Continuation of processing despite individual file failures

### Usability
- Comprehensive help documentation
- Intuitive command-line interface
- Clear logging with appropriate verbosity levels
- Preservation of original filenames in outputs

## Testing Considerations

### Unit Tests
- FileProcessor classification accuracy
- OutputManager path generation
- FPS measurement precision
- Error handling scenarios

### Integration Tests
- End-to-end processing workflows
- Multiple input format handling
- Output file verification
- Performance benchmarks

### Edge Cases
- Empty directories
- Unsupported file formats
- Insufficient disk space
- GPU memory limitations
- Invalid model config/checkpoint combinations

## Implementation Notes

### Code Organization
- Separate each major component into its own class
- Use composition over inheritance
- Implement clear interfaces between components
- Follow Python PEP 8 style guidelines

### Error Messages
- Provide actionable error messages
- Include file paths in error contexts
- Suggest solutions for common problems
- Log stack traces only in debug mode

### Performance Optimization
- Load model once and reuse for all inferences
- Use batch processing when possible
- Minimize memory allocations in loops
- Release resources promptly after use

This specification provides complete implementation guidance for developing the MMSegmentation inference script according to the discussed requirements and design decisions.