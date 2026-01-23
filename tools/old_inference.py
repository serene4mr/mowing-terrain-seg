#!/usr/bin/env python3
"""
MMSegmentation Custom Inference Script

A comprehensive inference script for MMSegmentation framework that handles multiple 
input types (images, videos, directories) with configurable output options and 
robust error handling.

Usage Examples:
    # Single image
    python inference.py --input demo.jpg --config configs/model.py --checkpoint model.pth --output-dir results/
    
    # Video with display and FPS overlay
    python inference.py --input video.mp4 --config configs/model.py --checkpoint model.pth --output-dir results/ --show --overlay-fps --opacity 0.6
    
    # Directory processing with masks
    python inference.py --input /path/to/data/ --config configs/model.py --checkpoint model.pth --output-dir results/ --save-masks --batch-size 4
    
    # Display only (no saving)
    python inference.py --input test_data/ --config configs/model.py --checkpoint model.pth --output-dir /tmp/unused --no-save --show --overlay-fps
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from mmseg.apis import init_model, inference_model

# Fix for PyTorch 2.6 weights_only loading issue
import os
import torch

# Set environment variable to disable weights_only for compatibility
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = 'False'

# Add src directory to Python path for custom datasets (similar to train.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all custom datasets to register them
import src  # This will import all datasets from src/__init__.py


def parse_args():
    """
    Parse command line arguments for MMSegmentation inference script.
    
    Returns:
        argparse.Namespace: Parsed arguments with validation
    """
    parser = argparse.ArgumentParser(
        description='MMSegmentation Custom Inference Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image processing
  python inference.py --input demo.jpg --config configs/model.py --checkpoint model.pth --output-dir results/
  
  # Video with display and FPS overlay
  python inference.py --input video.mp4 --config configs/model.py --checkpoint model.pth --output-dir results/ --show --overlay-fps --opacity 0.6
  
  # Directory processing with masks
  python inference.py --input /path/to/data/ --config configs/model.py --checkpoint model.pth --output-dir results/ --save-masks --batch-size 4
  
  # Display only (no saving)
  python inference.py --input test_data/ --config configs/model.py --checkpoint model.pth --output-dir /tmp/unused --no-save --show --overlay-fps
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input (image file, video file, or directory)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to model config file (.py)'
    )
    
    parser.add_argument(
        '--checkpoint', '-ckpt',
        type=str,
        required=True,
        help='Path to model checkpoint file (.pth)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        required=True,
        help='Directory to save all results'
    )
    
    # Optional arguments
    parser.add_argument(
        '--no-save',
        action='store_true',
        default=False,
        help='Disable saving any output files (default: False, saves overlays)'
    )
    
    parser.add_argument(
        '--save-masks',
        action='store_true',
        default=False,
        help='Save raw prediction masks (default: False)'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='Display results on-screen during processing (default: False)'
    )
    
    parser.add_argument(
        '--wait-time',
        type=int,
        default=1,
        help='Delay in ms between displayed frames when showing (default: 1)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Number of images to process per batch (default: 1)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Compute device (default: cuda:0)'
    )
    
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.7,
        help='Overlay opacity for visualization (0.0-1.0, default: 0.7)'
    )
    
    parser.add_argument(
        '--overlay-fps',
        action='store_true',
        default=False,
        help='Embed measured FPS text on result images/videos (default: False)'
    )
    
    parser.add_argument(
        '--conf-threshold',
        type=str,
        default=None,
        help='Confidence threshold(s) for filtering predictions. '
             'Single value (e.g., 0.8) applies to all classes. '
             'Comma-separated list (e.g., 0.9,0.7,0.8) applies per-class. '
             'Filtered pixels will be colored black.'
    )
    
    parser.add_argument(
        '--img-extensions',
        nargs='+',
        default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
        help='Image file extensions to process (default: .jpg .jpeg .png .bmp .tiff)'
    )
    
    parser.add_argument(
        '--vid-extensions',
        nargs='+',
        default=['.mp4', '.avi', '.mov', '.mkv', '.wmv'],
        help='Video file extensions to process (default: .mp4 .avi .mov .mkv .wmv)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    _validate_args(args)
    
    return args


def _validate_args(args):
    """
    Validate parsed arguments for correctness and consistency.
    
    Args:
        args (argparse.Namespace): Parsed arguments to validate
        
    Raises:
        ValueError: If validation fails
        FileNotFoundError: If required files don't exist
    """
    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {args.input}")
    
    # Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {args.config}")
    if not config_path.suffix == '.py':
        raise ValueError(f"Config file must be a Python file (.py): {args.config}")
    
    # Validate checkpoint file
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {args.checkpoint}")
    if not checkpoint_path.suffix == '.pth':
        raise ValueError(f"Checkpoint file must be a PyTorch file (.pth): {args.checkpoint}")
    
    # Validate output directory (create if doesn't exist)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate numeric arguments
    if args.batch_size < 1:
        raise ValueError(f"Batch size must be >= 1, got: {args.batch_size}")
    
    if not 0.0 <= args.opacity <= 1.0:
        raise ValueError(f"Opacity must be between 0.0 and 1.0, got: {args.opacity}")
    
    if args.wait_time < 0:
        raise ValueError(f"Wait time must be >= 0, got: {args.wait_time}")
    
    # Validate extensions
    for ext in args.img_extensions:
        if not ext.startswith('.'):
            raise ValueError(f"Image extension must start with '.', got: {ext}")
    
    for ext in args.vid_extensions:
        if not ext.startswith('.'):
            raise ValueError(f"Video extension must start with '.', got: {ext}")
    
    # Validate device
    if not args.device.startswith(('cuda:', 'cpu')):
        raise ValueError(f"Device must be 'cpu' or 'cuda:N', got: {args.device}")
    
    # Log validation success
    logging.info("Argument validation completed successfully")


def setup_logging():
    """
    Setup logging configuration with timestamps and INFO level.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


class FileProcessor:
    """
    FileProcessor class for input classification and file discovery.
    
    Handles single files, directories, and recursive scanning for supported
    image and video file types. Returns organized dictionary of discovered files.
    """
    
    def __init__(self, img_extensions: List[str], vid_extensions: List[str]):
        """
        Initialize FileProcessor with supported file extensions.
        
        Args:
            img_extensions (List[str]): List of supported image file extensions
            vid_extensions (List[str]): List of supported video file extensions
        """
        self.img_extensions = [ext.lower() for ext in img_extensions]
        self.vid_extensions = [ext.lower() for ext in vid_extensions]
        
        logging.info(f"FileProcessor initialized with image extensions: {self.img_extensions}")
        logging.info(f"FileProcessor initialized with video extensions: {self.vid_extensions}")
    
    def classify_input(self, input_path: Union[str, Path]) -> Dict[str, List[Path]]:
        """
        Classify input path and return organized dictionary of discovered files.
        
        Args:
            input_path (Union[str, Path]): Path to input (file or directory)
            
        Returns:
            Dict[str, List[Path]]: Dictionary with 'images' and 'videos' keys
                containing lists of discovered file paths
                
        Raises:
            FileNotFoundError: If input path doesn't exist
            ValueError: If input path is neither file nor directory
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        
        if input_path.is_file():
            return self._classify_single_file(input_path)
        elif input_path.is_dir():
            return self._classify_directory(input_path)
        else:
            raise ValueError(f"Input path is neither file nor directory: {input_path}")
    
    def _classify_single_file(self, file_path: Path) -> Dict[str, List[Path]]:
        """
        Classify a single file and return appropriate category.
        
        Args:
            file_path (Path): Path to the file to classify
            
        Returns:
            Dict[str, List[Path]]: Dictionary with file in appropriate category
        """
        file_ext = file_path.suffix.lower()
        
        if file_ext in self.img_extensions:
            logging.info(f"Classified as image: {file_path}")
            return {'images': [file_path], 'videos': []}
        elif file_ext in self.vid_extensions:
            logging.info(f"Classified as video: {file_path}")
            return {'images': [], 'videos': [file_path]}
        else:
            logging.warning(f"Unsupported file extension: {file_ext} for {file_path}")
            return {'images': [], 'videos': []}
    
    def _classify_directory(self, dir_path: Path) -> Dict[str, List[Path]]:
        """
        Recursively scan directory for supported files.
        
        Args:
            dir_path (Path): Path to directory to scan
            
        Returns:
            Dict[str, List[Path]]: Dictionary with discovered files organized by type
        """
        images = []
        videos = []
        
        try:
            # Recursively scan directory
            for file_path in dir_path.rglob('*'):
                if file_path.is_file():
                    file_ext = file_path.suffix.lower()
                    
                    if file_ext in self.img_extensions:
                        images.append(file_path)
                    elif file_ext in self.vid_extensions:
                        videos.append(file_path)
                    else:
                        # Skip unsupported files silently
                        continue
            
            # Sort paths for consistent ordering
            images.sort()
            videos.sort()
            
            logging.info(f"Discovered {len(images)} images and {len(videos)} videos in {dir_path}")
            
            return {'images': images, 'videos': videos}
            
        except PermissionError as e:
            logging.error(f"Permission denied accessing directory {dir_path}: {e}")
            raise
        except Exception as e:
            logging.error(f"Error scanning directory {dir_path}: {e}")
            raise
    
    def get_file_count(self, classification_result: Dict[str, List[Path]]) -> int:
        """
        Get total count of discovered files.
        
        Args:
            classification_result (Dict[str, List[Path]]): Result from classify_input
            
        Returns:
            int: Total number of discovered files
        """
        return len(classification_result['images']) + len(classification_result['videos'])
    
    def validate_files(self, classification_result: Dict[str, List[Path]]) -> Dict[str, List[Path]]:
        """
        Validate that discovered files are accessible and not corrupted.
        
        Args:
            classification_result (Dict[str, List[Path]]): Result from classify_input
            
        Returns:
            Dict[str, List[Path]]: Dictionary with only valid, accessible files
        """
        valid_images = []
        valid_videos = []
        
        # Validate images
        for img_path in classification_result['images']:
            if self._is_file_accessible(img_path):
                valid_images.append(img_path)
            else:
                logging.warning(f"Skipping inaccessible image: {img_path}")
        
        # Validate videos
        for vid_path in classification_result['videos']:
            if self._is_file_accessible(vid_path):
                valid_videos.append(vid_path)
            else:
                logging.warning(f"Skipping inaccessible video: {vid_path}")
        
        logging.info(f"File validation: {len(valid_images)} valid images, {len(valid_videos)} valid videos")
        
        return {'images': valid_images, 'videos': valid_videos}
    
    def _is_file_accessible(self, file_path: Path) -> bool:
        """
        Check if file is accessible and not corrupted.
        
        Args:
            file_path (Path): Path to file to check
            
        Returns:
            bool: True if file is accessible, False otherwise
        """
        try:
            # Check if file exists and is readable
            if not file_path.exists() or not file_path.is_file():
                return False
            
            # Check if file is readable
            if not os.access(file_path, os.R_OK):
                return False
            
            # Check file size (skip empty files)
            if file_path.stat().st_size == 0:
                logging.warning(f"Empty file detected: {file_path}")
                return False
            
            return True
            
        except (OSError, IOError) as e:
            logging.warning(f"Error checking file accessibility {file_path}: {e}")
            return False


class OutputManager:
    """
    OutputManager class for directory structure and filename generation.
    
    Handles creation of output directories, generation of timestamped filenames,
    and management of output file paths for both visualization overlays and masks.
    """
    
    def __init__(self, output_dir: Union[str, Path], save_masks: bool = False):
        """
        Initialize OutputManager with output directory and mask saving option.
        
        Args:
            output_dir (Union[str, Path]): Base output directory for all results
            save_masks (bool): Whether to save raw prediction masks
        """
        self.output_dir = Path(output_dir)
        self.save_masks = save_masks
        self.timestamp = self._generate_timestamp()
        
        # Create directory structure
        self._create_directories()
        
        logging.info(f"OutputManager initialized:")
        logging.info(f"  - Output directory: {self.output_dir}")
        logging.info(f"  - Save masks: {self.save_masks}")
        logging.info(f"  - Timestamp: {self.timestamp}")
    
    def _create_directories(self):
        """
        Create necessary output directory structure.
        """
        try:
            # Create main output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create masks subdirectory if needed
            if self.save_masks:
                self.masks_dir = self.output_dir / "masks"
                self.masks_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created masks directory: {self.masks_dir}")
            
            logging.info(f"Output directory structure created successfully")
            
        except Exception as e:
            logging.error(f"Error creating output directories: {e}")
            raise
    
    def _generate_timestamp(self) -> str:
        """
        Generate timestamp string for filename uniqueness.
        
        Returns:
            str: Timestamp in format YYYYMMDD_HHMMSS
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def get_output_paths(self, input_path: Union[str, Path], file_type: str) -> Tuple[Path, Path]:
        """
        Generate output paths for visualization overlay and optional mask.
        
        Args:
            input_path (Union[str, Path]): Path to input file
            file_type (str): Type of file ('image' or 'video')
            
        Returns:
            Tuple[Path, Path]: (overlay_path, mask_path) where mask_path is None if not saving masks
        """
        input_path = Path(input_path)
        
        # Generate base filename with timestamp
        stem = input_path.stem
        extension = input_path.suffix
        
        # Create overlay filename: <original_stem>_result_<timestamp>.<original_extension>
        overlay_filename = f"{stem}_result_{self.timestamp}{extension}"
        overlay_path = self.output_dir / overlay_filename
        
        # Create mask path if saving masks
        mask_path = None
        if self.save_masks:
            if file_type == 'image':
                # Image mask: <original_stem>_mask_<timestamp>.png
                mask_filename = f"{stem}_mask_{self.timestamp}.png"
                mask_path = self.masks_dir / mask_filename
            elif file_type == 'video':
                # Video frames: <original_stem>_frames_<timestamp>/frame_XXXXXX.png
                frames_dir = self.masks_dir / f"{stem}_frames_{self.timestamp}"
                frames_dir.mkdir(parents=True, exist_ok=True)
                # Return frames directory for video frame masks
                mask_path = frames_dir
        
        logging.debug(f"Generated output paths for {input_path}:")
        logging.debug(f"  - Overlay: {overlay_path}")
        logging.debug(f"  - Mask: {mask_path}")
        
        return overlay_path, mask_path
    
    def get_video_frame_mask_path(self, input_path: Union[str, Path], frame_number: int) -> Path:
        """
        Generate mask path for a specific video frame.
        
        Args:
            input_path (Union[str, Path]): Path to input video file
            frame_number (int): Frame number (0-indexed)
            
        Returns:
            Path: Path for the frame mask file
        """
        if not self.save_masks:
            raise ValueError("Cannot generate mask path when save_masks=False")
        
        input_path = Path(input_path)
        stem = input_path.stem
        
        # Create frames directory if it doesn't exist
        frames_dir = self.masks_dir / f"{stem}_frames_{self.timestamp}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate frame mask filename: frame_XXXXXX.png
        frame_filename = f"frame_{frame_number:06d}.png"
        frame_mask_path = frames_dir / frame_filename
        
        return frame_mask_path
    
    def ensure_output_directory(self, file_path: Path):
        """
        Ensure parent directory exists for output file.
        
        Args:
            file_path (Path): Path to output file
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error(f"Error creating directory for {file_path}: {e}")
            raise
    
    def get_output_info(self) -> Dict[str, Union[str, Path, bool]]:
        """
        Get information about output configuration.
        
        Returns:
            Dict[str, Union[str, Path, bool]]: Output configuration information
        """
        return {
            'output_dir': self.output_dir,
            'save_masks': self.save_masks,
            'timestamp': self.timestamp,
            'masks_dir': self.masks_dir if self.save_masks else None
        }
    
    def validate_output_paths(self, overlay_path: Path, mask_path: Path = None) -> bool:
        """
        Validate that output paths are accessible and writable.
        
        Args:
            overlay_path (Path): Path to overlay file
            mask_path (Path, optional): Path to mask file
            
        Returns:
            bool: True if all paths are valid, False otherwise
        """
        try:
            # Check overlay path
            if not self._is_path_writable(overlay_path):
                logging.error(f"Overlay path is not writable: {overlay_path}")
                return False
            
            # Check mask path if provided
            if mask_path is not None and not self._is_path_writable(mask_path):
                logging.error(f"Mask path is not writable: {mask_path}")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating output paths: {e}")
            return False
    
    def _is_path_writable(self, file_path: Path) -> bool:
        """
        Check if a path is writable.
        
        Args:
            file_path (Path): Path to check
            
        Returns:
            bool: True if path is writable, False otherwise
        """
        try:
            # Ensure parent directory exists
            self.ensure_output_directory(file_path)
            
            # Check if we can write to the directory
            test_file = file_path.parent / f".test_write_{self.timestamp}"
            test_file.touch()
            test_file.unlink()  # Clean up test file
            
            return True
            
        except Exception as e:
            logging.warning(f"Path not writable {file_path}: {e}")
            return False


class MMSegInference:
    """
    MMSegInference class for model operations and FPS measurement.
    
    Handles model loading, inference operations, FPS measurement, and result processing
    for both single images, batch images, and videos.
    """
    
    def __init__(self, config_path: Union[str, Path], checkpoint_path: Union[str, Path], device: str = 'cuda:0'):
        """
        Initialize MMSegInference with model configuration and checkpoint.
        
        Args:
            config_path (Union[str, Path]): Path to model config file
            checkpoint_path (Union[str, Path]): Path to model checkpoint file
            device (str): Compute device for inference
        """
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        
        # Initialize model
        self.model = None
        self._load_model()
        
        logging.info(f"MMSegInference initialized:")
        logging.info(f"  - Config: {self.config_path}")
        logging.info(f"  - Checkpoint: {self.checkpoint_path}")
        logging.info(f"  - Device: {self.device}")
    
    def _load_model(self):
        """
        Load and initialize the MMSegmentation model.
        """
        try:
            # Validate model configuration
            if not self.validate_model_config():
                raise ValueError("Model configuration validation failed")
            
            # Temporarily monkey patch torch.load to use weights_only=False
            import torch
            original_load = torch.load
            
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            torch.load = patched_load
            
            # Initialize model
            self.model = init_model(
                str(self.config_path),
                str(self.checkpoint_path),
                device=self.device
            )
            
            # Restore original torch.load
            torch.load = original_load
            
            logging.info("Model loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    
    def validate_model_config(self) -> bool:
        """
        Validate model configuration and checkpoint compatibility.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            # Check if config file exists and is readable
            if not self.config_path.exists():
                logging.error(f"Config file does not exist: {self.config_path}")
                return False
            
            # Check if checkpoint file exists and is readable
            if not self.checkpoint_path.exists():
                logging.error(f"Checkpoint file does not exist: {self.checkpoint_path}")
                return False
            
            # Basic file size checks
            if self.config_path.stat().st_size == 0:
                logging.error(f"Config file is empty: {self.config_path}")
                return False
            
            if self.checkpoint_path.stat().st_size == 0:
                logging.error(f"Checkpoint file is empty: {self.checkpoint_path}")
                return False
            
            logging.info("Model configuration validation passed")
            return True
            
        except Exception as e:
            logging.error(f"Error validating model configuration: {e}")
            return False
    
    def process_image(self, img_path: Union[str, Path], output_manager, **kwargs) -> Dict:
        """
        Process a single image for inference.
        
        Args:
            img_path (Union[str, Path]): Path to input image
            output_manager: OutputManager instance for path generation
            **kwargs: Additional arguments (show, overlay_fps, opacity, etc.)
            
        Returns:
            Dict: Processing results with paths and metadata
        """
        img_path = Path(img_path)
        start_time = time.perf_counter()
        
        try:
            # Generate output paths
            overlay_path, mask_path = output_manager.get_output_paths(img_path, 'image')
            
            # Perform inference
            result = inference_model(self.model, str(img_path))
            
            # Measure FPS
            inference_time = time.perf_counter() - start_time
            fps = 1.0 / inference_time if inference_time > 0 else 0.0
            
            # Get dataset info for custom colormap
            dataset_info = get_dataset_info(self.config_path)
            custom_colormap = None
            if dataset_info['palette']:
                custom_colormap = create_custom_colormap(dataset_info['palette'], dataset_info['num_classes'])
            
            # Load original image for overlay
            original_image = cv2.imread(str(img_path))
            if original_image is None:
                raise ValueError(f"Could not load image: {img_path}")
            
            # Get segmentation mask from result
            if hasattr(result, 'pred_sem_seg') and result.pred_sem_seg is not None:
                mask = result.pred_sem_seg.data.cpu().numpy()
                if len(mask.shape) == 3:
                    mask = mask[0]  # Remove batch dimension if present
            else:
                raise ValueError("No segmentation mask found in result")
            
            # Apply confidence threshold if specified
            if kwargs.get('conf_thresholds') is not None:
                # Get confidence scores from seg_logits
                if hasattr(result, 'seg_logits') and result.seg_logits is not None:
                    # Extract logits and convert to probabilities
                    import torch
                    import torch.nn.functional as F
                    
                    logits = result.seg_logits.data.cpu().numpy()
                    # logits shape is [num_classes, height, width] - no batch dimension
                    
                    # Convert to probabilities using softmax
                    logits_tensor = torch.from_numpy(logits)
                    probs = F.softmax(logits_tensor, dim=0).numpy()
                    confidence_scores = np.max(probs, axis=0)
                    
                    # Apply confidence threshold
                    mask = apply_confidence_threshold(mask, confidence_scores, kwargs['conf_thresholds'])
                    logging.info(f"Applied confidence thresholding. Confidence range: {confidence_scores.min():.4f} to {confidence_scores.max():.4f}")
                else:
                    logging.warning("No seg_logits found in result. Skipping confidence filtering.")
            
            # Save segmentation result
            save_success = save_segmentation_result(
                original_image, mask, overlay_path,
                save_mask=output_manager.save_masks,
                mask_path=mask_path,
                custom_colormap=custom_colormap,
                opacity=kwargs.get('opacity', 0.7)
            )
            
            if not save_success:
                raise ValueError(f"Failed to save segmentation result: {overlay_path}")
            
            # Process results
            processing_result = {
                'input_path': img_path,
                'overlay_path': overlay_path,
                'mask_path': mask_path,
                'fps': fps,
                'inference_time': inference_time,
                'result': result,
                'success': True
            }
            
            logging.info(f"Image processed successfully: {img_path}")
            logging.info(f"  - FPS: {fps:.2f}")
            logging.info(f"  - Inference time: {inference_time:.4f}s")
            logging.info(f"  - Saved overlay: {overlay_path}")
            if output_manager.save_masks:
                logging.info(f"  - Saved mask: {mask_path}")
            
            return processing_result
            
        except Exception as e:
            logging.error(f"Error processing image {img_path}: {e}")
            return {
                'input_path': img_path,
                'overlay_path': None,
                'mask_path': None,
                'fps': 0.0,
                'inference_time': 0.0,
                'result': None,
                'success': False,
                'error': str(e)
            }
    
    def process_images_batch(self, img_paths: List[Union[str, Path]], output_manager, **kwargs) -> List[Dict]:
        """
        Process multiple images in batch for inference.
        
        Args:
            img_paths (List[Union[str, Path]]): List of input image paths
            output_manager: OutputManager instance for path generation
            **kwargs: Additional arguments (show, overlay_fps, opacity, etc.)
            
        Returns:
            List[Dict]: List of processing results for each image
        """
        results = []
        batch_start_time = time.perf_counter()
        
        logging.info(f"Processing batch of {len(img_paths)} images")
        
        for i, img_path in enumerate(img_paths):
            logging.info(f"Processing image {i+1}/{len(img_paths)}: {img_path}")
            
            # Process individual image
            result = self.process_image(img_path, output_manager, **kwargs)
            results.append(result)
        
        # Calculate batch statistics
        batch_time = time.perf_counter() - batch_start_time
        successful_results = [r for r in results if r['success']]
        avg_fps = np.mean([r['fps'] for r in successful_results]) if successful_results else 0.0
        
        logging.info(f"Batch processing completed:")
        logging.info(f"  - Total images: {len(img_paths)}")
        logging.info(f"  - Successful: {len(successful_results)}")
        logging.info(f"  - Failed: {len(img_paths) - len(successful_results)}")
        logging.info(f"  - Batch time: {batch_time:.2f}s")
        logging.info(f"  - Average FPS: {avg_fps:.2f}")
        
        return results
    
    def process_video(self, vid_path: Union[str, Path], output_manager, **kwargs) -> Dict:
        """
        Process a video file frame by frame for inference.
        
        Args:
            vid_path (Union[str, Path]): Path to input video file
            output_manager: OutputManager instance for path generation
            **kwargs: Additional arguments (show, overlay_fps, opacity, wait_time, etc.)
            
        Returns:
            Dict: Processing results with paths and metadata
        """
        vid_path = Path(vid_path)
        start_time = time.perf_counter()
        
        try:
            # Generate output paths
            overlay_path, mask_path = output_manager.get_output_paths(vid_path, 'video')
            
            # Open video
            cap = cv2.VideoCapture(str(vid_path))
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {vid_path}")
            
            # Get video properties
            fps_original = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logging.info(f"Video properties:")
            logging.info(f"  - FPS: {fps_original}")
            logging.info(f"  - Frame count: {frame_count}")
            logging.info(f"  - Resolution: {width}x{height}")
            
            # Setup video writer for output
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(overlay_path), fourcc, fps_original, (width, height))
            
            # Process frames
            frame_times = []
            processed_frames = 0
            
            for frame_idx in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame_start = time.perf_counter()
                result = inference_model(self.model, frame)
                frame_time = time.perf_counter() - frame_start
                frame_times.append(frame_time)
                
                # Get segmentation mask from result
                if hasattr(result, 'pred_sem_seg') and result.pred_sem_seg is not None:
                    mask = result.pred_sem_seg.data.cpu().numpy()
                    if len(mask.shape) == 3:
                        mask = mask[0]  # Remove batch dimension if present
                    
                    # Apply confidence threshold if specified
                    if kwargs.get('conf_thresholds') is not None:
                        # Get confidence scores from seg_logits
                        if hasattr(result, 'seg_logits') and result.seg_logits is not None:
                            # Extract logits and convert to probabilities
                            import torch
                            import torch.nn.functional as F
                            
                            logits = result.seg_logits.data.cpu().numpy()
                            # logits shape is [num_classes, height, width] - no batch dimension
                            
                            # Convert to probabilities using softmax
                            logits_tensor = torch.from_numpy(logits)
                            probs = F.softmax(logits_tensor, dim=0).numpy()
                            confidence_scores = np.max(probs, axis=0)
                            
                            # Apply confidence threshold
                            mask = apply_confidence_threshold(mask, confidence_scores, kwargs['conf_thresholds'])
                        else:
                            logging.warning("No seg_logits found in result. Skipping confidence filtering.")
                    
                    # Get dataset info for custom colormap
                    dataset_info = get_dataset_info(self.config_path)
                    custom_colormap = None
                    if dataset_info['palette']:
                        custom_colormap = create_custom_colormap(dataset_info['palette'], dataset_info['num_classes'])
                    
                    # Create segmentation overlay
                    frame = create_visualization_overlay(
                        frame, mask, 
                        opacity=kwargs.get('opacity', 0.7),
                        custom_colormap=custom_colormap
                    )
                
                # Apply FPS overlay if needed
                if kwargs.get('overlay_fps', False):
                    frame_fps = 1.0 / frame_time if frame_time > 0 else 0.0
                    frame = self._overlay_fps_on_frame(frame, frame_fps)
                
                # Write frame to output video
                out.write(frame)
                
                # Save mask if needed
                if output_manager.save_masks and mask_path:
                    frame_mask_path = output_manager.get_video_frame_mask_path(vid_path, frame_idx)
                    # Save mask (implementation depends on result format)
                    # This would need to be implemented based on the actual result format
                
                processed_frames += 1
                
                # Show frame if requested
                if kwargs.get('show', False):
                    cv2.imshow('Inference', frame)
                    wait_time = kwargs.get('wait_time', 1)
                    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                        break
            
            # Cleanup
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # Calculate statistics
            total_time = time.perf_counter() - start_time
            avg_fps = processed_frames / total_time if total_time > 0 else 0.0
            avg_frame_time = np.mean(frame_times) if frame_times else 0.0
            
            processing_result = {
                'input_path': vid_path,
                'overlay_path': overlay_path,
                'mask_path': mask_path,
                'fps': avg_fps,
                'inference_time': total_time,
                'processed_frames': processed_frames,
                'total_frames': frame_count,
                'avg_frame_time': avg_frame_time,
                'success': True
            }
            
            logging.info(f"Video processed successfully: {vid_path}")
            logging.info(f"  - Processed frames: {processed_frames}/{frame_count}")
            logging.info(f"  - Average FPS: {avg_fps:.2f}")
            logging.info(f"  - Total time: {total_time:.2f}s")
            
            return processing_result
            
        except Exception as e:
            logging.error(f"Error processing video {vid_path}: {e}")
            return {
                'input_path': vid_path,
                'overlay_path': None,
                'mask_path': None,
                'fps': 0.0,
                'inference_time': 0.0,
                'processed_frames': 0,
                'total_frames': 0,
                'avg_frame_time': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def _measure_fps(self, start_time: float, count: int) -> float:
        """
        Measure FPS based on start time and frame count.
        
        Args:
            start_time (float): Start time from time.perf_counter()
            count (int): Number of frames processed
            
        Returns:
            float: FPS value
        """
        elapsed = time.perf_counter() - start_time
        return count / elapsed if elapsed > 0 else 0.0
    
    def _overlay_fps_on_frame(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """
        Overlay FPS text on frame.
        
        Args:
            frame (np.ndarray): Input frame
            fps (float): FPS value to display
            
        Returns:
            np.ndarray: Frame with FPS overlay
        """
        try:
            # Create a copy to avoid modifying original
            frame_with_fps = frame.copy()
            
            # FPS text
            fps_text = f"FPS: {fps:.1f}"
            
            # Text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1.0
            color = (0, 255, 0)  # Green
            thickness = 2
            position = (10, 30)
            
            # Add text to frame
            cv2.putText(frame_with_fps, fps_text, position, font, scale, color, thickness)
            
            return frame_with_fps
            
        except Exception as e:
            logging.warning(f"Error overlaying FPS on frame: {e}")
            return frame


# Utility Functions
def measure_inference_fps(start_time: float, frame_count: int) -> float:
    """
    Measure FPS based on start time and frame count.
    
    Args:
        start_time (float): Start time from time.perf_counter()
        frame_count (int): Number of frames processed
        
    Returns:
        float: FPS value
    """
    elapsed = time.perf_counter() - start_time
    return frame_count / elapsed if elapsed > 0 else 0.0


def overlay_fps_on_frame(frame: np.ndarray, fps: float, position: tuple = (10, 30), 
                        font: int = cv2.FONT_HERSHEY_SIMPLEX, scale: float = 1.0, 
                        color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """
    Overlay FPS text on frame with customizable properties.
    
    Args:
        frame (np.ndarray): Input frame
        fps (float): FPS value to display
        position (tuple): Text position (x, y)
        font (int): OpenCV font type
        scale (float): Text scale factor
        color (tuple): Text color (B, G, R)
        thickness (int): Text thickness
        
    Returns:
        np.ndarray: Frame with FPS overlay
    """
    try:
        # Create a copy to avoid modifying original
        frame_with_fps = frame.copy()
        
        # FPS text
        fps_text = f"FPS: {fps:.1f}"
        
        # Add text to frame
        cv2.putText(frame_with_fps, fps_text, position, font, scale, color, thickness)
        
        return frame_with_fps
        
    except Exception as e:
        logging.warning(f"Error overlaying FPS on frame: {e}")
        return frame


def validate_config_file(config_path: Union[str, Path]) -> bool:
    """
    Validate model configuration file.
    
    Args:
        config_path (Union[str, Path]): Path to config file
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        config_path = Path(config_path)
        
        # Check if file exists
        if not config_path.exists():
            logging.error(f"Config file does not exist: {config_path}")
            return False
        
        # Check if it's a Python file
        if not config_path.suffix == '.py':
            logging.error(f"Config file must be a Python file (.py): {config_path}")
            return False
        
        # Check file size
        if config_path.stat().st_size == 0:
            logging.error(f"Config file is empty: {config_path}")
            return False
        
        # Try to import the config (basic syntax check)
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", str(config_path))
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            logging.info(f"Config file syntax validation passed: {config_path}")
            return True
        except Exception as e:
            logging.error(f"Config file syntax error: {e}")
            return False
            
    except Exception as e:
        logging.error(f"Error validating config file: {e}")
        return False


def validate_checkpoint_file(checkpoint_path: Union[str, Path]) -> bool:
    """
    Validate model checkpoint file.
    
    Args:
        checkpoint_path (Union[str, Path]): Path to checkpoint file
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        checkpoint_path = Path(checkpoint_path)
        
        # Check if file exists
        if not checkpoint_path.exists():
            logging.error(f"Checkpoint file does not exist: {checkpoint_path}")
            return False
        
        # Check if it's a PyTorch file
        if not checkpoint_path.suffix == '.pth':
            logging.error(f"Checkpoint file must be a PyTorch file (.pth): {checkpoint_path}")
            return False
        
        # Check file size
        if checkpoint_path.stat().st_size == 0:
            logging.error(f"Checkpoint file is empty: {checkpoint_path}")
            return False
        
        # Try to load the checkpoint (basic format check)
        try:
            import torch
            # Use weights_only=False for compatibility with older checkpoints
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
            
            # Check if it has expected keys
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint or 'model' in checkpoint:
                    logging.info(f"Checkpoint file format validation passed: {checkpoint_path}")
                    return True
                else:
                    logging.warning(f"Checkpoint file may not contain model weights: {checkpoint_path}")
                    return True  # Still valid, just different format
            else:
                logging.info(f"Checkpoint file format validation passed: {checkpoint_path}")
                return True
                
        except Exception as e:
            logging.error(f"Checkpoint file format error: {e}")
            return False
            
    except Exception as e:
        logging.error(f"Error validating checkpoint file: {e}")
        return False


def apply_confidence_threshold(mask: np.ndarray, confidence_scores: np.ndarray, 
                              thresholds: List[float]) -> np.ndarray:
    """
    Apply confidence threshold filtering to segmentation mask.
    
    Args:
        mask (np.ndarray): Segmentation mask
        confidence_scores (np.ndarray): Confidence scores for each pixel
        thresholds (List[float]): Confidence thresholds per class
        
    Returns:
        np.ndarray: Filtered mask with black pixels for low-confidence areas
    """
    if thresholds is None:
        return mask
    
    # Create filtered mask
    filtered_mask = mask.copy()
    
    # Apply threshold for each class
    for class_id, threshold in enumerate(thresholds):
        class_mask = (mask == class_id)
        low_confidence = (confidence_scores < threshold) & class_mask
        filtered_mask[low_confidence] = -1  # Mark for black color
    
    return filtered_mask


def create_visualization_overlay(image: np.ndarray, mask: np.ndarray, 
                                opacity: float = 0.7, colormap: int = cv2.COLORMAP_JET,
                                custom_colormap: np.ndarray = None) -> np.ndarray:
    """
    Create visualization overlay with proper opacity blending.
    
    Args:
        image (np.ndarray): Original image
        mask (np.ndarray): Segmentation mask
        opacity (float): Overlay opacity (0.0-1.0)
        colormap (int): OpenCV colormap for mask visualization
        custom_colormap (np.ndarray): Custom colormap from dataset palette
        
    Returns:
        np.ndarray: Blended overlay image
    """
    try:
        # Ensure mask is in correct format
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        # Create colored mask
        if custom_colormap is not None:
            # Use custom colormap from dataset
            colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            for class_id in range(len(custom_colormap)):
                class_mask = (mask == class_id)
                colored_mask[class_mask] = custom_colormap[class_id]
        else:
            # Use default OpenCV colormap
            colored_mask = cv2.applyColorMap(mask, colormap)
        
        # Resize colored mask to match image if needed
        if colored_mask.shape[:2] != image.shape[:2]:
            colored_mask = cv2.resize(colored_mask, (image.shape[1], image.shape[0]))
        
        # Blend with proper opacity using cv2.addWeighted
        overlay = cv2.addWeighted(image, 1 - opacity, colored_mask, opacity, 0)
        
        return overlay
        
    except Exception as e:
        logging.warning(f"Error creating visualization overlay: {e}")
        return image


def save_segmentation_result(image: np.ndarray, mask: np.ndarray, 
                           output_path: Union[str, Path], 
                           save_mask: bool = False, mask_path: Union[str, Path] = None,
                           custom_colormap: np.ndarray = None, opacity: float = 0.7) -> bool:
    """
    Save segmentation result (image with overlay and optional mask).
    
    Args:
        image (np.ndarray): Original image
        mask (np.ndarray): Segmentation mask
        output_path (Union[str, Path]): Path to save overlay image
        save_mask (bool): Whether to save raw mask
        mask_path (Union[str, Path]): Path to save raw mask
        custom_colormap (np.ndarray): Custom colormap from dataset palette
        opacity (float): Overlay opacity (0.0-1.0)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        output_path = Path(output_path)
        
        # Create visualization overlay with custom colormap and opacity
        overlay = create_visualization_overlay(image, mask, opacity=opacity, custom_colormap=custom_colormap)
        
        # Save overlay image
        success = cv2.imwrite(str(output_path), overlay)
        if not success:
            logging.error(f"Failed to save overlay image: {output_path}")
            return False
        
        # Save raw mask if requested
        if save_mask and mask_path:
            mask_path = Path(mask_path)
            mask_success = cv2.imwrite(str(mask_path), mask)
            if not mask_success:
                logging.warning(f"Failed to save mask: {mask_path}")
            else:
                logging.info(f"Saved mask: {mask_path}")
        
        logging.info(f"Saved result: {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error saving segmentation result: {e}")
        return False


def get_model_info(config_path: Union[str, Path], checkpoint_path: Union[str, Path]) -> Dict:
    """
    Get model information from config and checkpoint files.
    
    Args:
        config_path (Union[str, Path]): Path to config file
        checkpoint_path (Union[str, Path]): Path to checkpoint file
        
    Returns:
        Dict: Model information
    """
    try:
        info = {
            'config_path': str(config_path),
            'checkpoint_path': str(checkpoint_path),
            'config_size': Path(config_path).stat().st_size if Path(config_path).exists() else 0,
            'checkpoint_size': Path(checkpoint_path).stat().st_size if Path(checkpoint_path).exists() else 0,
            'config_valid': validate_config_file(config_path),
            'checkpoint_valid': validate_checkpoint_file(checkpoint_path)
        }
        
        # Try to get additional info from checkpoint
        try:
            import torch
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict):
                info['checkpoint_keys'] = list(checkpoint.keys())
                if 'meta' in checkpoint:
                    info['meta'] = checkpoint['meta']
        except Exception as e:
            logging.warning(f"Could not extract additional checkpoint info: {e}")
        
        return info
        
    except Exception as e:
        logging.error(f"Error getting model info: {e}")
        return {
            'config_path': str(config_path),
            'checkpoint_path': str(checkpoint_path),
            'error': str(e)
        }


def setup_progress_bar(total: int, desc: str = "Processing") -> object:
    """
    Setup progress bar for processing.
    
    Args:
        total (int): Total number of items to process
        desc (str): Description for progress bar
        
    Returns:
        object: Progress bar object
    """
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit="item")
    except ImportError:
        logging.warning("tqdm not available, using simple progress logging")
        return None


def log_processing_stats(results: List[Dict], processing_type: str = "processing") -> None:
    """
    Log comprehensive processing statistics.
    
    Args:
        results (List[Dict]): List of processing results
        processing_type (str): Type of processing (images, videos, etc.)
    """
    try:
        if not results:
            logging.info(f"No {processing_type} results to log")
            return
        
        # Calculate statistics
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        total_count = len(results)
        success_count = len(successful)
        failure_count = len(failed)
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        
        # Log basic statistics
        logging.info(f"{processing_type.title()} Statistics:")
        logging.info(f"  - Total: {total_count}")
        logging.info(f"  - Successful: {success_count}")
        logging.info(f"  - Failed: {failure_count}")
        logging.info(f"  - Success Rate: {success_rate:.1f}%")
        
        # Log performance statistics if available
        if successful:
            fps_values = [r.get('fps', 0) for r in successful if 'fps' in r]
            if fps_values:
                avg_fps = np.mean(fps_values)
                max_fps = np.max(fps_values)
                min_fps = np.min(fps_values)
                logging.info(f"  - Average FPS: {avg_fps:.2f}")
                logging.info(f"  - Max FPS: {max_fps:.2f}")
                logging.info(f"  - Min FPS: {min_fps:.2f}")
        
        # Log timing statistics if available
        timing_values = [r.get('inference_time', 0) for r in successful if 'inference_time' in r]
        if timing_values:
            avg_time = np.mean(timing_values)
            total_time = np.sum(timing_values)
            logging.info(f"  - Average Time: {avg_time:.4f}s")
            logging.info(f"  - Total Time: {total_time:.2f}s")
        
        # Log errors if any
        if failed:
            logging.warning(f"Failed {processing_type}:")
            for i, result in enumerate(failed[:5]):  # Log first 5 failures
                error_msg = result.get('error', 'Unknown error')
                input_path = result.get('input_path', 'Unknown path')
                logging.warning(f"  - {input_path}: {error_msg}")
            
            if len(failed) > 5:
                logging.warning(f"  - ... and {len(failed) - 5} more failures")
        
    except Exception as e:
        logging.error(f"Error logging processing stats: {e}")


def validate_conf_threshold(conf_threshold_str: str, num_classes: int) -> List[float]:
    """
    Validate and parse confidence threshold argument.
    
    Args:
        conf_threshold_str (str): Comma-separated threshold values
        num_classes (int): Number of classes in dataset
        
    Returns:
        List[float]: Parsed threshold values
        
    Raises:
        ValueError: If threshold format is invalid
    """
    if conf_threshold_str is None:
        return None
    
    try:
        # Parse comma-separated values
        conf_thresholds = [float(x.strip()) for x in conf_threshold_str.split(',')]
        
        # Validate threshold values
        for threshold in conf_thresholds:
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {threshold}")
        
        # Validate list length
        if len(conf_thresholds) == 1:
            # Apply to all classes
            return [conf_thresholds[0]] * num_classes
        elif len(conf_thresholds) == num_classes:
            # Apply per-class
            return conf_thresholds
        else:
            raise ValueError(f"Confidence threshold list length ({len(conf_thresholds)}) must be 1 or equal to number of classes ({num_classes})")
            
    except ValueError as e:
        if "could not convert" in str(e):
            raise ValueError(f"Invalid confidence threshold format: {conf_threshold_str}. Use comma-separated numbers (e.g., 0.8 or 0.9,0.7,0.8)")
        else:
            raise e


def get_dataset_info(config_path: Union[str, Path]) -> Dict:
    """
    Get dataset information directly from config file.
    
    Args:
        config_path (Union[str, Path]): Path to config file
        
    Returns:
        Dict: Dataset information including classes and palette
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", str(config_path))
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Get metainfo directly from config
    metainfo = getattr(config_module, 'metainfo', None)
    
    if metainfo:
        return {
            'classes': metainfo.get('classes'),
            'palette': metainfo.get('palette'),
            'num_classes': len(metainfo.get('classes', []))
        }
    else:
        # Try to get from base config if this config uses _base_ imports
        try:
            base_config_path = 'configs/_base_/datasets/ycor-lm-3cls.py'
            base_spec = importlib.util.spec_from_file_location("base_config", base_config_path)
            base_config_module = importlib.util.module_from_spec(base_spec)
            base_spec.loader.exec_module(base_config_module)
            
            base_metainfo = getattr(base_config_module, 'metainfo', None)
            if base_metainfo:
                return {
                    'classes': base_metainfo.get('classes'),
                    'palette': base_metainfo.get('palette'),
                    'num_classes': len(base_metainfo.get('classes', []))
                }
        except Exception as e:
            logging.warning(f"Could not load base config: {e}")
        
        return {
            'classes': None,
            'palette': None,
            'num_classes': 0
        }


def create_custom_colormap(palette: List[List[int]], num_classes: int) -> np.ndarray:
    """
    Create a custom colormap from dataset palette.
    
    Args:
        palette (List[List[int]]): Color palette from dataset
        num_classes (int): Number of classes
        
    Returns:
        np.ndarray: Custom colormap array
    """
    try:
        # Create colormap array
        colormap = np.zeros((num_classes, 3), dtype=np.uint8)
        
        for i, color in enumerate(palette):
            if i < num_classes:
                # Convert RGB to BGR for OpenCV
                colormap[i] = [color[2], color[1], color[0]]  # RGB -> BGR
        
        return colormap
        
    except Exception as e:
        logging.warning(f"Error creating custom colormap: {e}")
        # Fallback to default colormap
        return np.array([[128, 128, 128], [0, 255, 0], [255, 0, 0]], dtype=np.uint8)


def apply_custom_colormap(mask: np.ndarray, colormap: np.ndarray) -> np.ndarray:
    """
    Apply custom colormap to segmentation mask.
    
    Args:
        mask (np.ndarray): Segmentation mask
        colormap (np.ndarray): Custom colormap
        
    Returns:
        np.ndarray: Colored mask
    """
    try:
        # Ensure mask is in correct format
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        # Apply custom colormap
        colored_mask = colormap[mask]
        
        return colored_mask
        
    except Exception as e:
        logging.warning(f"Error applying custom colormap: {e}")
        # Fallback to default colormap
        return cv2.applyColorMap(mask, cv2.COLORMAP_JET)


def main():
    """
    Main entry point for the inference script.
    """
    # Setup logging
    setup_logging()
    
    try:
        # Parse arguments
        args = parse_args()
        
        # Log successful argument parsing
        logging.info("Starting MMSegmentation inference script")
        logging.info(f"Input: {args.input}")
        logging.info(f"Config: {args.config}")
        logging.info(f"Checkpoint: {args.checkpoint}")
        logging.info(f"Output directory: {args.output_dir}")
        logging.info(f"Device: {args.device}")
        logging.info(f"Batch size: {args.batch_size}")
        logging.info(f"Save masks: {args.save_masks}")
        logging.info(f"Show results: {args.show}")
        logging.info(f"Overlay FPS: {args.overlay_fps}")
        
        # Initialize components
        logging.info("Initializing inference components...")
        
        # Create FileProcessor instance
        file_processor = FileProcessor(args.img_extensions, args.vid_extensions)
        
        # Create OutputManager instance
        output_manager = OutputManager(args.output_dir, args.save_masks)
        
        # Validate confidence thresholds
        conf_thresholds = None
        if args.conf_threshold is not None:
            dataset_info = get_dataset_info(args.config)
            num_classes = dataset_info['num_classes']
            conf_thresholds = validate_conf_threshold(args.conf_threshold, num_classes)
            logging.info(f"Confidence thresholds: {conf_thresholds}")
        
        # Create MMSegInference instance
        try:
            mmseg_inference = MMSegInference(args.config, args.checkpoint, args.device)
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            logging.error("Please check your config and checkpoint files")
            sys.exit(1)
        
        # Process input files
        logging.info("Starting file processing...")
        
        # Classify input
        classification_result = file_processor.classify_input(args.input)
        
        # Validate files
        valid_files = file_processor.validate_files(classification_result)
        
        # Log discovered files
        total_files = file_processor.get_file_count(valid_files)
        logging.info(f"Discovered {total_files} files to process:")
        logging.info(f"  - Images: {len(valid_files['images'])}")
        logging.info(f"  - Videos: {len(valid_files['videos'])}")
        
        if total_files == 0:
            logging.warning("No valid files found to process")
            return
        
        # Process images
        if valid_files['images']:
            logging.info(f"Processing {len(valid_files['images'])} images...")
            
            if args.batch_size > 1:
                # Batch processing
                logging.info(f"Using batch processing with batch size: {args.batch_size}")
                results = mmseg_inference.process_images_batch(
                    valid_files['images'], 
                    output_manager,
                    show=args.show,
                    overlay_fps=args.overlay_fps,
                    opacity=args.opacity,
                    conf_thresholds=conf_thresholds
                )
            else:
                # Single image processing
                results = []
                for i, img_path in enumerate(valid_files['images']):
                    logging.info(f"Processing image {i+1}/{len(valid_files['images'])}: {img_path}")
                    result = mmseg_inference.process_image(
                        img_path, 
                        output_manager,
                        show=args.show,
                        overlay_fps=args.overlay_fps,
                        opacity=args.opacity,
                        conf_thresholds=conf_thresholds
                    )
                    results.append(result)
            
            # Log image processing results
            successful_images = [r for r in results if r['success']]
            failed_images = [r for r in results if not r['success']]
            
            logging.info(f"Image processing completed:")
            logging.info(f"  - Successful: {len(successful_images)}")
            logging.info(f"  - Failed: {len(failed_images)}")
            
            if successful_images:
                avg_fps = np.mean([r['fps'] for r in successful_images])
                logging.info(f"  - Average FPS: {avg_fps:.2f}")
        
        # Process videos
        if valid_files['videos']:
            logging.info(f"Processing {len(valid_files['videos'])} videos...")
            
            for i, vid_path in enumerate(valid_files['videos']):
                logging.info(f"Processing video {i+1}/{len(valid_files['videos'])}: {vid_path}")
                
                result = mmseg_inference.process_video(
                    vid_path, 
                    output_manager,
                    show=args.show,
                    overlay_fps=args.overlay_fps,
                    wait_time=args.wait_time,
                    conf_thresholds=conf_thresholds
                )
                
                if result['success']:
                    logging.info(f"Video processing successful:")
                    logging.info(f"  - Processed frames: {result['processed_frames']}/{result['total_frames']}")
                    logging.info(f"  - Average FPS: {result['fps']:.2f}")
                    logging.info(f"  - Total time: {result['inference_time']:.2f}s")
                else:
                    logging.error(f"Video processing failed: {result.get('error', 'Unknown error')}")
        
        # Final summary
        logging.info("Inference processing completed successfully!")
        logging.info(f"Results saved to: {args.output_dir}")
        
        if args.save_masks:
            logging.info(f"Masks saved to: {output_manager.masks_dir}")
        
        if args.show:
            logging.info("Press 'q' to quit if display windows are open")
        
    except Exception as e:
        logging.error(f"Error in argument parsing: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
