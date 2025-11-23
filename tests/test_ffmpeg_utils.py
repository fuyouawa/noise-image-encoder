"""
Unit tests for FFmpeg-specific utilities.
"""

import unittest

# Import the video utilities
from utils.video.ffmpeg import (
    get_video_format,
    list_available_formats
)


class TestFFmpegUtilities(unittest.TestCase):
    """Test FFmpeg-specific utilities."""

    def test_get_video_format(self):
        """Test video format configuration retrieval."""
        # Test existing formats
        av1_format = get_video_format("av1-webm")
        self.assertIn("main_pass", av1_format)
        self.assertEqual(av1_format["extension"], "webm")

        h264_format = get_video_format("h264-mp4")
        self.assertIn("main_pass", h264_format)
        self.assertEqual(h264_format["extension"], "mp4")

        # Test non-existent format
        with self.assertRaises(KeyError):
            get_video_format("non-existent-format")

    def test_list_available_formats(self):
        """Test listing available video formats."""
        formats = list_available_formats()

        self.assertIsInstance(formats, list)
        self.assertIn("av1-webm", formats)
        self.assertIn("h264-mp4", formats)
        self.assertIn("h265-mp4", formats)
        self.assertIn("webm", formats)


if __name__ == "__main__":
    unittest.main()