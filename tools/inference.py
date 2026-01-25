import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.mowing_terrain_seg.inference.predictor import SegPredictor, Backend
from src.mowing_terrain_seg.utils.logger import LOGGER


def parse_args():
    parser = argparse.ArgumentParser(description='Mowing Terrain Segmentation Inference')
    
    # Model arguments
    parser.add_argument('--cfg-uri', '--cfg', dest='cfg_uri', required=True, 
                        help='Path to necessary config (.py or pipeline.json)')
    parser.add_argument('--model-uri', '--model', dest='model_uri', required=True, 
                        help='Path to model (e.g., .pth, .onnx, or .engine)')
    parser.add_argument('--input', '-i', required=True, help='Path to input image/video/directory')
    parser.add_argument('--output-dir', '-o', default='results/', help='Directory to save results')

    # Predictor settings
    parser.add_argument('--backend', '-b', type=str, default='torch', 
                        choices=['torch', 'onnx', 'tensorrt'], help='Inference backend')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--conf-threshold', type=str, default=None,
                        help='Confidence threshold (float or comma-separated list)')

    # Visualization/Output settings
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--opacity', type=float, default=0.7, help='Overlay opacity (0-1)')
    parser.add_argument('--show', action='store_true', help='Show results in a window')
    parser.add_argument('--save-vis', action='store_true', default=True, help='Save visualization results (default: True)')
    parser.add_argument('--no-save-vis', dest='save_vis', action='store_false', help='Do not save visualization results')
    parser.add_argument('--save-mask', action='store_true', help='Save raw prediction masks (.png)')
    parser.add_argument('--overlay-fps', action='store_true', help='Overlay FPS on visualization')

    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1. Setup Output Directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(args.output_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info(f"Results will be saved to: {output_path}")
    
    # 2. Parse confidence thresholds
    conf_thresholds = None
    if args.conf_threshold:
        try:
            if ',' in args.conf_threshold:
                conf_thresholds = [float(x.strip()) for x in args.conf_threshold.split(',')]
            else:
                conf_thresholds = float(args.conf_threshold)
        except ValueError:
            LOGGER.error(f"Invalid confidence threshold format: {args.conf_threshold}")
            sys.exit(1)

    # 3. Initialize Predictor
    LOGGER.info(f"Initializing predictor with backend: {args.backend}")
    try:
        predictor = SegPredictor(
            cfg_uri=args.cfg_uri,
            model_uri=args.model_uri,
            backend=Backend(args.backend),
            device=args.device,
            conf_thresholds=conf_thresholds
        )
    except Exception as e:
        LOGGER.error(f"Failed to initialize predictor: {e}")
        sys.exit(1)
    
    LOGGER.info("Predictor initialized successfully.")
    
    # TODO: Phase 2 - Implement Input Loader and Processing Loop

if __name__ == "__main__":
    main()
