"""InferenceSource type detection."""
import os
import tempfile

import pytest

from mowing_terrain_seg.inference.source import InferenceSource, SourceType


def test_camera_id_int():
    s = InferenceSource(0, batch_size=1)
    assert s.type == SourceType.CAMERA_ID


def test_camera_id_str_digit():
    s = InferenceSource("0", batch_size=1)
    assert s.type == SourceType.CAMERA_ID


def test_stream_url():
    s = InferenceSource("rtsp://192.168.0.1/stream", batch_size=1)
    assert s.type == SourceType.STREAM_URL


def test_image_dir():
    with tempfile.TemporaryDirectory() as d:
        s = InferenceSource(d, batch_size=1)
        assert s.type == SourceType.IMAGE_DIR


def test_image_file():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.jpg")
        with open(p, "wb") as f:
            f.write(b"")
        s = InferenceSource(p, batch_size=1)
        assert s.type == SourceType.IMAGE_FILE


def test_video_file():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.mp4")
        with open(p, "wb") as f:
            f.write(b"")
        s = InferenceSource(p, batch_size=1)
        assert s.type == SourceType.VIDEO_FILE


def test_unknown_raises():
    with pytest.raises(ValueError, match="Could not detect"):
        InferenceSource("/nonexistent/path/that/is/not/file/or/dir/xyz123", batch_size=1)
