import os
import cv2
import numpy as np
from typing import Union, Optional, List, Dict, Any, Tuple
from enum import Enum

class SourceType(Enum):
    """
    An enumeration of the possible source types.
    """
    IMAGE_FILE = "image_file"
    IMAGE_DIR = "image_dir"
    VIDEO_FILE = "video_file"
    VIDEO_DIR = "video_dir"
    CAMERA_ID = "camera_id"
    STREAM_URL = "stream_url"

class InferenceSource:
    """
    A class to handle the inference source.
    """
    
    def __init__(self, src: Union[str, int], batch_size: int = 1):
        """
        Initialize the inference source with automatic type detection.
        
        Args:
            src: Path to a file, a directory, a camera ID, or a stream URL.
            batch_size: Number of frames to yield per iteration.
        """
        self.src = src
        self.batch_size = batch_size
        
        # Internal auto-detection (The "Brain")
        self.type = self._detect_type(src)
        
        # Placeholders for handlers
        self.cap: Optional[cv2.VideoCapture] = None
        self.file_list: List[str] = []
        self.total_count: int = 0
        self.current_idx: int = 0
        
    def _detect_type(self, src: Union[str, int]) -> SourceType:
        """
        Identify the source type based on input pattern and filesystem checks.
        """
        # 1. Camera ID (Integer or digit string)
        if isinstance(src, int) or (isinstance(src, str) and src.isdigit()):
            return SourceType.CAMERA_ID

        # 2. Stream URL (Protocols)
        if isinstance(src, str) and src.startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
            return SourceType.STREAM_URL

        # 3. Directories
        if os.path.isdir(str(src)):
            # Logic: If it's a directory, check if it contains primarily videos or images
            files = [f.lower() for f in os.listdir(src)]
            video_exts = ('.mp4', '.avi', '.mkv', '.mov')
            
            # If the directory contains video files, we might treat it as a VIDEO_DIR
            if any(f.endswith(video_exts) for f in files):
                return SourceType.VIDEO_DIR
            return SourceType.IMAGE_DIR

        # 4. Files
        if os.path.isfile(str(src)):
            src_str = str(src).lower()
            video_exts = ('.mp4', '.avi', '.mkv', '.mov', '.wmv')
            image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
            
            if src_str.endswith(video_exts):
                return SourceType.VIDEO_FILE
            if src_str.endswith(image_exts):
                return SourceType.IMAGE_FILE

        raise ValueError(f"Could not detect source type for: {src}")
    
    def _setup_source(self):
        """
        Initialize the data handlers (file list or video capture) based on source type.
        """
        if self.type == SourceType.IMAGE_FILE:
            self.file_list = [str(self.src)]
            self.total_count = 1

        elif self.type == SourceType.IMAGE_DIR:
            extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
            self.file_list = sorted([
                os.path.join(self.src, f) for f in os.listdir(self.src)
                if f.lower().endswith(extensions)
            ])
            self.total_count = len(self.file_list)

        elif self.type in [SourceType.VIDEO_FILE, SourceType.CAMERA_ID, SourceType.STREAM_URL]:
            # Handle int conversion for local camera ID
            src = int(self.src) if self.type == SourceType.CAMERA_ID else self.src
            self.cap = cv2.VideoCapture(src)
            
            if not self.cap.isOpened():
                raise IOError(f"Failed to open {self.type.value}: {src}")
            
            if self.type == SourceType.VIDEO_FILE:
                self.total_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                # Streams/Cameras are effectively infinite
                self.total_count = -1 

        elif self.type == SourceType.VIDEO_DIR:
            # Handle directory containing video files (process first video found for now)
            video_exts = ('.mp4', '.avi', '.mkv', '.mov')
            videos = sorted([
                os.path.join(self.src, f) for f in os.listdir(self.src)
                if f.lower().endswith(video_exts)
            ])
            if not videos:
                raise FileNotFoundError(f"No video files found in {self.src}")
            self.cap = cv2.VideoCapture(videos[0])
            self.total_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Release resources (video capture handles)."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def is_stream(self) -> bool:
        """Check if source is a live stream or camera."""
        return self.type in [SourceType.CAMERA_ID, SourceType.STREAM_URL]

    @property
    def fps(self) -> float:
        """Get source FPS (returns 30.0 for streams by default)."""
        if self.cap is not None:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return fps if fps > 0 else 30.0
        return 30.0

    def __iter__(self):
        self._setup_source()
        self.current_idx = 0
        return self

    def __next__(self) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Get the next batch of images and metadata.
        
        Returns:
            Tuple: (List of image numpy arrays, List of metadata dictionaries)
        """
        batch_imgs = []
        batch_metas = []
        
        for _ in range(self.batch_size):
            # 1. Check for termination (finite sources)
            if self.total_count != -1 and self.current_idx >= self.total_count:
                break

            img = None
            meta = {}
            
            # 2. Read from Disk (Images)
            if self.type in [SourceType.IMAGE_FILE, SourceType.IMAGE_DIR]:
                if self.current_idx < len(self.file_list):
                    img_path = self.file_list[self.current_idx]
                    img = cv2.imread(img_path)
                    if img is None:
                        # Skip corrupted images
                        self.current_idx += 1
                        continue
                        
                    meta = {
                        'path': img_path,
                        'name': os.path.basename(img_path),
                        'index': self.current_idx,
                        'source_type': self.type.value
                    }
                else:
                    break

            # 3. Read from Capture (Video/Stream/Camera)
            elif self.type in [SourceType.VIDEO_FILE, SourceType.VIDEO_DIR, SourceType.CAMERA_ID, SourceType.STREAM_URL]:
                if self.cap is not None:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    img = frame
                    
                    # For video, we generate a name based on the index
                    name = f"frame_{self.current_idx:06d}.jpg"
                    if self.type == SourceType.VIDEO_FILE:
                        # Try to get the original video name
                        video_name = os.path.basename(str(self.src)).split('.')[0]
                        name = f"{video_name}_{name}"

                    meta = {
                        'name': name,
                        'index': self.current_idx,
                        'timestamp_ms': self.cap.get(cv2.CAP_PROP_POS_MSEC),
                        'source_type': self.type.value
                    }
                else:
                    break
            
            if img is not None:
                batch_imgs.append(img)
                batch_metas.append(meta)
                self.current_idx += 1
            else:
                break
        
        # 4. If we couldn't collect ANY images, the source is exhausted
        if not batch_imgs:
            raise StopIteration
            
        return batch_imgs, batch_metas


        