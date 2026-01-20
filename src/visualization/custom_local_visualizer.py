from typing import Optional, Dict, List
import os

import cv2
import mmcv
import numpy as np

from mmengine.dist import master_only
from mmengine.structures import PixelData

from mmseg.visualization import SegLocalVisualizer
from mmseg.registry import VISUALIZERS
from mmseg.structures import SegDataSample


@VISUALIZERS.register_module()
class CustomSegLocalVisualizer(SegLocalVisualizer):
    """Custom segmentation visualizer with 4-panel layout support.
    
    Extends the standard MMSegmentation visualizer to provide:
    - Custom horizontal layout: Original → GT → Prediction → Confidence
    - Confidence map visualization with VIRIDIS colormap
    - Flexible image saving with iteration-based organization
    - User-controllable visualization components
    
    The visualizer creates a horizontal layout where each panel is optional
    based on the draw_* parameters. Confidence maps show model uncertainty
    using a heatmap overlay (green = high confidence, purple = low confidence).
    """
    
    def __init__(self, 
                 save_interval: int = 5, 
                 max_images_per_iter: int = 5,
                 name: str = 'custom_seg_local_visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[dict] = None,
                 save_dir: Optional[str] = None,
                 classes: Optional[list] = None,
                 palette: Optional[list] = None,
                 dataset_name: Optional[str] = None,
                 alpha: float = 0.8,
                 **kwargs):
        """Initialize the custom segmentation visualizer.
        
        Args:
            save_interval (int): Interval for saving validation images. Defaults to 5.
            max_images_per_iter (int): Maximum number of images to save per iteration. Defaults to 5.
            name (str): Name of the visualizer. Defaults to 'custom_seg_local_visualizer'.
            image (np.ndarray, optional): Initial image. Defaults to None.
            vis_backends (dict, optional): Visualization backends. Defaults to None.
            save_dir (str, optional): Directory to save visualizations. Defaults to None.
            classes (list, optional): Class names for segmentation. Defaults to None.
            palette (list, optional): Color palette for classes. Defaults to None.
            dataset_name (str, optional): Name of the dataset. Defaults to None.
            alpha (float): Transparency for overlays. Defaults to 0.8.
            **kwargs: Additional arguments passed to parent class.
        """
        # Store custom parameters for image saving control
        self.save_interval = save_interval
        self.max_images_per_iter = max_images_per_iter
        self.val_image_counter = 0
        self._last_iter = None
        self.save_dir = save_dir
        
        # Initialize parent visualizer with standard MMSeg parameters
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir,
            classes=classes,
            palette=palette,
            dataset_name=dataset_name,
            alpha=alpha,
            **kwargs
        )
        
    @master_only
    def add_datasample(self,
                      name: str,
                      image: np.ndarray,
                      data_sample: Optional[SegDataSample] = None,
                      draw_gt: bool = True,
                      draw_pred: bool = True,
                      show: bool = False,
                      wait_time: float = 0,
                      step: int = 0,
                      with_labels: Optional[bool] = True,
                      draw_ori: bool = True,
                      draw_conf: bool = True) -> None:
        """Add a data sample to the visualizer with custom layout support.
        
        This method handles the MMSegmentation interface and file management,
        then delegates to the internal visualization method.
        
        Args:
            name (str): Name of the data sample.
            image (np.ndarray): Input image array.
            data_sample (SegDataSample, optional): Segmentation data sample.
            draw_gt (bool): Whether to draw ground truth. Defaults to True.
            draw_pred (bool): Whether to draw prediction. Defaults to True.
            show (bool): Whether to show the image. Defaults to False.
            wait_time (float): Wait time for display. Defaults to 0.
            step (int): Current step number. Defaults to 0.
            with_labels (bool): Whether to show labels. Defaults to True.
            draw_ori (bool): Whether to draw original image. Defaults to True.
            draw_conf (bool): Whether to draw confidence map. Defaults to True.
        """
        
        # Extract iteration from name (e.g., "val_rgb.jpg_4000" -> 4000)
        if '_' in name:
            try:
                current_iter = int(name.split('_')[-1])
            except ValueError:
                current_iter = step
        else:
            current_iter = step
            
        # Reset counter for new iteration
        if self._last_iter != current_iter:
            self.val_image_counter = 0
            self._last_iter = current_iter
            
        # Increment counter for each validation image
        self.val_image_counter += 1
        
        # Check if we should save this image
        should_save = (
            self.val_image_counter % self.save_interval == 0 and
            self.val_image_counter <= self.max_images_per_iter * self.save_interval
        )
        
        if should_save:
            # Create unique name with folder structure: iter_4000/val_rgb_10
            new_name = f"iter_{current_iter}/val_rgb_{self.val_image_counter:02d}"
            
            # Use the visualizer's save_dir

            # Create the proper path: save_dir/vis_data/vis_image/iter_X/
            vis_image_dir = os.path.join(self.save_dir, 'vis_data', 'vis_image')
            iter_dir = os.path.join(vis_image_dir, f"iter_{current_iter}")
            os.makedirs(iter_dir, exist_ok=True)
            
            # Define out_file path
            out_file = os.path.join(iter_dir, f'val_rgb_{self.val_image_counter:02d}.png')
        
            # Call parent method
            # super().add_datasample(
            #     name=new_name,
            #     image=image,
            #     data_sample=data_sample,
            #     draw_gt=draw_gt,
            #     draw_pred=draw_pred,
            #     show=show,
            #     wait_time=wait_time,
            #     out_file=out_file,
            #     step=step,
            #     with_labels=with_labels
            # )
            self._visualize_datasample(
                name=new_name,
                image=image,
                data_sample=data_sample,
                draw_gt=draw_gt,
                draw_pred=draw_pred,
                show=show,
                wait_time=wait_time,
                out_file=out_file,
                step=step,
                with_labels=with_labels,
                draw_ori=draw_ori,
                draw_conf=draw_conf
            )
            
    def _visualize_datasample(self,
                             name: str,
                             image: np.ndarray,
                             data_sample: Optional[SegDataSample] = None,
                             draw_gt: bool = True,
                             draw_pred: bool = True,
                             show: bool = False,
                             wait_time: float = 0,
                             out_file: Optional[str] = None,
                             step: int = 0,
                             with_labels: Optional[bool] = True,
                             draw_ori: bool = True,
                             draw_conf: bool = True) -> None:
        """Create visualization with custom 4-panel layout.
        
        Creates a horizontal layout with: Original → GT → Prediction → Confidence
        Each panel is optional based on the draw_* parameters.
        
        Args:
            name (str): Name for the visualization.
            image (np.ndarray): Input image array.
            data_sample (SegDataSample, optional): Segmentation data sample.
            draw_gt (bool): Whether to draw ground truth. Defaults to True.
            draw_pred (bool): Whether to draw prediction. Defaults to True.
            show (bool): Whether to show the image. Defaults to False.
            wait_time (float): Wait time for display. Defaults to 0.
            out_file (str, optional): Output file path. Defaults to None.
            step (int): Current step number. Defaults to 0.
            with_labels (bool): Whether to show labels. Defaults to True.
            draw_ori (bool): Whether to draw original image. Defaults to True.
            draw_conf (bool): Whether to draw confidence map. Defaults to True.
        """
        
        # Get dataset metadata for visualization
        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)
    
        # Initialize all image data containers
        ori_img_data = None
        gt_img_data = None
        pred_img_data = None
        conf_img_data = None
        
        # 1. Original image (always available)
        if draw_ori:
            ori_img_data = image
    
        # 2. Ground Truth segmentation
        if draw_gt and data_sample is not None:
            if 'gt_sem_seg' in data_sample:
                assert classes is not None, 'class information is not provided'
                gt_img_data = self._draw_sem_seg(image, data_sample.gt_sem_seg,
                                                 classes, palette, with_labels)
    
        # 3. Prediction segmentation
        if draw_pred and data_sample is not None:
            if 'pred_sem_seg' in data_sample:
                assert classes is not None, 'class information is not provided'
                pred_img_data = self._draw_sem_seg(image, data_sample.pred_sem_seg,
                                                   classes, palette, with_labels)
    
        # 4. Confidence map (requires seg_logits)
        if draw_conf and data_sample is not None:
            if 'seg_logits' in data_sample:
                conf_img_data = self._draw_conf_map(image, data_sample.seg_logits)
    
        # Create horizontal layout: Original → GT → Prediction → Confidence
        images_to_combine = []
        
        # Add images in exact order (maintains consistent layout)
        if ori_img_data is not None:
            images_to_combine.append(ori_img_data)
        if gt_img_data is not None:
            images_to_combine.append(gt_img_data)
        if pred_img_data is not None:
            images_to_combine.append(pred_img_data)
        if conf_img_data is not None:
            images_to_combine.append(conf_img_data)
        
        # Combine all images horizontally
        if len(images_to_combine) > 1:
            drawn_img = np.concatenate(images_to_combine, axis=1)
        elif len(images_to_combine) == 1:
            drawn_img = images_to_combine[0]
        else:
            drawn_img = image  # Fallback to original image
    
        # Display or save the result
        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)
    
        if out_file is not None:
            mmcv.imwrite(mmcv.rgb2bgr(drawn_img), out_file)
        else:
            self.add_image(name, drawn_img, step)
            
    def _draw_conf_map(self, 
                       image: np.ndarray, 
                       seg_logits: PixelData) -> np.ndarray:
        """Draw confidence map overlay using VIRIDIS colormap.
        
        Creates a heatmap where:
        - Purple/Blue = Low confidence
        - Green/Yellow = High confidence
        
        Args:
            image (np.ndarray): The original image to draw on.
            seg_logits (PixelData): Raw model logits for confidence extraction.

        Returns:
            np.ndarray: The drawn image with confidence overlay (RGB).
        """
        # Extract logits from PixelData
        logits = seg_logits.data
        if hasattr(logits, 'cpu'):
            logits = logits.cpu().numpy()

        # Ensure logits are in (C, H, W) format for proper processing
        if logits.ndim == 3:
            if logits.shape[-1] == logits.shape[0]:  # (H, W, C) -> (C, H, W)
                logits = logits.transpose(2, 0, 1)

        # Apply softmax to convert logits to probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)

        # Get maximum probability (confidence) for each pixel
        confidence = np.max(probs, axis=0)

        # Ensure confidence values are in [0, 1] range
        confidence = np.clip(confidence, 0, 1)

        # Convert to 0-255 range for colormap visualization
        confidence_vis = (confidence * 255).astype(np.uint8)

        # Apply VIRIDIS colormap (green for high confidence)
        confidence_colored = cv2.applyColorMap(confidence_vis, cv2.COLORMAP_VIRIDIS)

        # Convert BGR to RGB for proper display
        confidence_colored = cv2.cvtColor(confidence_colored, cv2.COLOR_BGR2RGB)

        # Blend with original image (60% confidence overlay, 40% original)
        blended = cv2.addWeighted(image, 0.4, confidence_colored, 0.6, 0)

        return blended

