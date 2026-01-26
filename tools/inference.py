import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mowing_terrain_seg.inference import SegPredictor, Backend, InferenceSource, SourceType, InferenceTimer
from src.mowing_terrain_seg.utils.logger import LOGGER


def parse_args():
    parser = argparse.ArgumentParser(description='Mowing Terrain Segmentation Inference')
    
    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--cfg-uri', '-c', required=True, 
                             help='Path to model config file (.py or pipeline.json)')
    model_group.add_argument('--model-uri', '-m', required=True, 
                             help='Path to model weights/checkpoint file (.pth, .onnx, or .engine)')
    model_group.add_argument('--backend', '-b', type=str, default='torch', 
                        choices=['torch', 'onnx', 'tensorrt'], help='Inference backend')
    model_group.add_argument('--device', default='cuda:0', help='Device used for inference')

    # Data I/O arguments
    data_group = parser.add_argument_group('Data I/O')
    data_group.add_argument('--input', '-i', required=True, 
                            help='Path to input image/video/directory/camera_id/stream_url')
    data_group.add_argument('--output-dir', '-o', default='work_dirs/inference', 
                            help='Root directory to save results')
    data_group.add_argument('--batch-size', type=int, default=1, help='Batch size for inference')

    # Predictor settings
    logic_group = parser.add_argument_group('Inference Logic')
    logic_group.add_argument('--conf-threshold', type=float, nargs='+', default=None,
                        help='Confidence threshold (single float or per-class list)')

    # Visualization/Output settings
    vis_group = parser.add_argument_group('Visualization & Export')
    vis_group.add_argument('--show', action='store_true', help='Show results in a window')
    vis_group.add_argument('--opacity', type=float, default=0.7, help='Overlay opacity (0-1)')
    vis_group.add_argument('--save-vis', action='store_true', help='Save visualized results')
    vis_group.add_argument('--save-mask', action='store_true', help='Save raw prediction masks as .png')
    vis_group.add_argument('--overlay-fps', action='store_true', help='Draw FPS overlay on results')

    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1. Initialize Run Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    vis_dir = os.path.join(run_dir, 'vis')
    mask_dir = os.path.join(run_dir, 'masks')
    
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)
    if args.save_mask:
        os.makedirs(mask_dir, exist_ok=True)
        
    LOGGER.info(f"Initialized inference run: {run_dir}")
    
    # 2. Initialize Predictor
    predictor = SegPredictor(
        cfg_uri=args.cfg_uri,
        model_uri=args.model_uri,
        backend=Backend(args.backend),
        device=args.device,
        conf_thresholds=args.conf_threshold
    )
    
    # 3. Initialize Source
    source = InferenceSource(src=args.input, batch_size=args.batch_size)
    timer = InferenceTimer(device=args.device)
    
    # Generate auto-palette for visualization
    auto_palette = predictor.get_auto_palette()
    
    LOGGER.info(f"Source type: {source.type.value}, Batch size: {args.batch_size}")
    
    # Placeholder for video writer
    video_writer = None
    is_temporal = source.type in [
        SourceType.VIDEO_FILE, SourceType.VIDEO_DIR, 
        SourceType.CAMERA_ID, SourceType.STREAM_URL
    ]

    # 4. Main Inference Loop
    try:
        for imgs, metas in tqdm(source, desc="Inference"):
            t0 = timer.tick()
            
            # Predict handles pre-process, forward, and post-process
            masks = predictor.predict(imgs)
            
            t1 = timer.tick()
            # Average inference time per image in the batch
            infer_time_per_img = (t1 - t0) / len(imgs) if imgs else 0
            
            # B. Visualization and Saving
            t2 = timer.tick()
            for i, (img, mask, meta) in enumerate(zip(imgs, masks, metas)):
                vis_img = predictor.visualize_mask(img, mask, opacity=args.opacity, palette=auto_palette)
                
                if args.overlay_fps:
                    avg_fps = timer.get_avg_fps()
                    # If first frame, use current inference time to estimate FPS
                    fps = avg_fps if avg_fps > 0 else (1.0 / infer_time_per_img if infer_time_per_img > 0 else 0)
                    cv2.putText(vis_img, f"FPS: {fps:.1f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if args.show:
                    cv2.imshow('Mowing Terrain Segmentation', vis_img)
                    is_static = source.type in [SourceType.IMAGE_FILE, SourceType.IMAGE_DIR]
                    wait_time = 0 if is_static else 1
                    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                        LOGGER.info("Quit by user ('q' pressed).")
                        return

                # --- Saving Logic ---
                filename = meta['name']
                
                # Save Raw Mask
                if args.save_mask:
                    mask_path = os.path.join(mask_dir, filename.replace('.jpg', '.png'))
                    cv2.imwrite(mask_path, mask)
                
                # Save Visualization
                if args.save_vis:
                    if is_temporal:
                        # Video/Stream: Save to a single video file
                        if video_writer is None:
                            h, w = vis_img.shape[:2]
                            video_name = f"result_{timestamp}.mp4"
                            video_path = os.path.join(vis_dir, video_name)
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(video_path, fourcc, source.fps, (w, h))
                            LOGGER.info(f"Initialized VideoWriter: {video_path}")
                        
                        video_writer.write(vis_img)
                    else:
                        # Images: Save individual files
                        vis_path = os.path.join(vis_dir, filename)
                        cv2.imwrite(vis_path, vis_img)
                
            t3 = timer.tick()
            post_time_per_img = (t3 - t2) / len(imgs) if imgs else 0
            
            # Record timing per image (approximate)
            for _ in range(len(imgs)):
                timer.record(0, infer_time_per_img, post_time_per_img)

    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    finally:
        if video_writer:
            video_writer.release()
            LOGGER.info("VideoWriter released.")
        source.close()
        cv2.destroyAllWindows()
        
    # 5. Performance Reporting
    stats = timer.get_stats()
    if stats:
        report = f"\n" + "="*50 + "\n"
        report += f"INFERENCE PERFORMANCE REPORT\n"
        report += "="*50 + "\n"
        report += f"Backend:      {args.backend.upper()}\n"
        report += f"Device:       {args.device}\n"
        report += f"Total Frames: {len(timer.total_times)}\n"
        report += "-"*50 + "\n"
        report += f"Avg FPS:      {stats['avg_fps']:.2f}\n"
        report += f"Avg Latency:  {stats['avg_total']:.2f} ms\n"
        report += f"  - Pre-proc: {stats['avg_pre']:.2f} ms (included in infer)\n"
        report += f"  - Infer:    {stats['avg_infer']:.2f} ms\n"
        report += f"  - Post-proc: {stats['avg_post']:.2f} ms\n"
        report += f"P99 Latency:  {stats['p99_latency']:.2f} ms\n"
        report += "="*50 + "\n"
        LOGGER.info(report)
        
        # Save stats to JSON
        import json
        with open(os.path.join(run_dir, 'performance.json'), 'w') as f:
            json.dump(stats, f, indent=4)

if __name__ == "__main__":
    main()
