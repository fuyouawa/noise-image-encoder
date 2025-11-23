"""
Unit tests for VideoInfo class.
"""

import unittest
import numpy as np

# Import the video utilities
from utils.video.common import VideoInfo


class TestVideoInfo(unittest.TestCase):
    """Test VideoInfo class functionality."""

    def test_video_info_creation(self):
        """Test VideoInfo object creation and basic properties."""
        info = VideoInfo(
            source_path="test.mp4",
            source_fps=16.0,
            source_width=1920,
            source_height=1080,
            loaded_width=720,
            loaded_height=1280,
            loaded_channels=3,
            source_frame_count=97,
            loaded_frame_count=97,
            loaded_fps=15.0,
            generator="ffmpeg"
        )

        self.assertEqual(info.source_path, "test.mp4")
        self.assertEqual(info.source_fps, 16.0)
        self.assertEqual(info.source_width, 1920)
        self.assertEqual(info.source_height, 1080)
        self.assertEqual(info.loaded_width, 720)
        self.assertEqual(info.loaded_height, 1280)
        self.assertEqual(info.loaded_channels, 3)
        self.assertEqual(info.loaded_frame_count, 97)
        self.assertEqual(info.loaded_fps, 15.0)
        self.assertEqual(info.total_duration, 6.0625)
        self.assertEqual(info.generator, "ffmpeg")

    def test_video_info_properties(self):
        """Test VideoInfo computed properties."""
        info = VideoInfo(
            source_path="test.mp4",
            source_fps=16.0,
            source_width=1920,
            source_height=1080,
            loaded_width=720,
            loaded_height=1280,
            loaded_channels=3,
            source_frame_count=97,
            loaded_frame_count=97,
            loaded_fps=15.0,
            generator="ffmpeg"
        )

        self.assertEqual(info.resolution, "720x1280")
        self.assertAlmostEqual(info.aspect_ratio, 720/1280)
        self.assertAlmostEqual(info.estimated_duration, 6.0625)

    def test_video_info_string_representations(self):
        """Test VideoInfo string and repr methods."""
        info = VideoInfo(
            source_path="test.mp4",
            source_fps=16.0,
            source_width=1920,
            source_height=1080,
            loaded_width=720,
            loaded_height=1280,
            loaded_channels=3,
            source_frame_count=97,
            loaded_frame_count=97,
            loaded_fps=15.0,
            generator="ffmpeg"
        )

        self.assertIn("test.mp4", str(info))
        self.assertIn("720x1280", str(info))
        self.assertIn("source fps: 16.0fps", str(info))
        self.assertIn("loaded fps: 15.0fps", str(info))
        self.assertIn("source: 1920x1080", str(info))
        self.assertIn("VideoInfo", repr(info))
        self.assertIn("source_width=1920", repr(info))
        self.assertIn("source_height=1080", repr(info))
        self.assertIn("loaded_fps=15.0", repr(info))


if __name__ == "__main__":
    unittest.main()