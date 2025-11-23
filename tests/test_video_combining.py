"""
Unit tests for video combining functionality.
"""

import unittest
import tempfile
import os
from PIL import Image
import numpy as np

# Import the video utilities
from utils.video.ffmpeg import combine_video as ffmpeg_combine_video
from utils.video.opencv import combine_video as opencv_combine_video
from utils.video.ffmpeg import get_video_format, list_available_formats


class TestVideoCombining(unittest.TestCase):
    """Test video combining functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test frames
        self.test_frames = []
        for i in range(5):
            # Create simple colored frames
            color = (i * 50, i * 50, i * 50)
            img = Image.new("RGB", (100, 100), color=color)
            self.test_frames.append(img)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_ffmpeg_combine_video_basic(self):
        """Test basic FFmpeg video combining."""
        output_path = os.path.join(self.temp_dir, "test_ffmpeg.mp4")

        result_path = ffmpeg_combine_video(
            image_batch=self.test_frames,
            output_path=output_path,
            frame_rate=10,
            video_format="video/h264-mp4"
        )

        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_ffmpeg_combine_video_with_pingpong(self):
        """Test FFmpeg video combining with pingpong effect."""
        output_path = os.path.join(self.temp_dir, "test_ffmpeg_pingpong.mp4")

        result_path = ffmpeg_combine_video(
            image_batch=self.test_frames,
            output_path=output_path,
            frame_rate=10,
            video_format="video/h264-mp4",
            pingpong=True
        )

        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_ffmpeg_combine_video_with_metadata(self):
        """Test FFmpeg video combining with metadata."""
        output_path = os.path.join(self.temp_dir, "test_ffmpeg_metadata.mp4")

        video_metadata = {
            "title": "Test Video",
            "artist": "Test Artist",
            "description": "A test video created for unit testing"
        }

        result_path = ffmpeg_combine_video(
            image_batch=self.test_frames,
            output_path=output_path,
            frame_rate=10,
            video_format="video/h264-mp4",
            video_metadata=video_metadata
        )

        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_ffmpeg_combine_video_different_formats(self):
        """Test FFmpeg video combining with different video formats."""
        available_formats = list_available_formats()

        for format_name in available_formats:
            with self.subTest(format=format_name):
                output_path = os.path.join(self.temp_dir, f"test_ffmpeg_{format_name}.{get_video_format(format_name)['extension']}")

                try:
                    result_path = ffmpeg_combine_video(
                        image_batch=self.test_frames,
                        output_path=output_path,
                        frame_rate=10,
                        video_format=f"video/{format_name}"
                    )

                    self.assertEqual(result_path, output_path)
                    self.assertTrue(os.path.exists(output_path))
                    self.assertGreater(os.path.getsize(output_path), 0)
                except Exception as e:
                    # Some formats might not be available on the system
                    self.skipTest(f"Format {format_name} not available: {e}")

    def test_ffmpeg_combine_video_invalid_format(self):
        """Test FFmpeg video combining with invalid format."""
        output_path = os.path.join(self.temp_dir, "test_invalid.mp4")

        with self.assertRaises(KeyError):
            ffmpeg_combine_video(
                image_batch=self.test_frames,
                output_path=output_path,
                frame_rate=10,
                video_format="video/invalid-format"
            )

    def test_opencv_combine_video_basic(self):
        """Test basic OpenCV video combining."""
        output_path = os.path.join(self.temp_dir, "test_opencv.mp4")

        result_path = opencv_combine_video(
            image_batch=self.test_frames,
            output_path=output_path,
            frame_rate=10,
            video_format="video/mp4"
        )

        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_opencv_combine_video_with_pingpong(self):
        """Test OpenCV video combining with pingpong effect."""
        output_path = os.path.join(self.temp_dir, "test_opencv_pingpong.mp4")

        result_path = opencv_combine_video(
            image_batch=self.test_frames,
            output_path=output_path,
            frame_rate=10,
            video_format="video/mp4",
            pingpong=True
        )

        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_opencv_combine_video_different_formats(self):
        """Test OpenCV video combining with different video formats."""
        test_formats = ["mp4", "avi", "mov"]

        for format_ext in test_formats:
            with self.subTest(format=format_ext):
                output_path = os.path.join(self.temp_dir, f"test_opencv.{format_ext}")

                try:
                    result_path = opencv_combine_video(
                        image_batch=self.test_frames,
                        output_path=output_path,
                        frame_rate=10,
                        video_format=f"video/{format_ext}"
                    )

                    self.assertEqual(result_path, output_path)
                    self.assertTrue(os.path.exists(output_path))
                    self.assertGreater(os.path.getsize(output_path), 0)
                except Exception as e:
                    # Some formats might not be available on the system
                    self.skipTest(f"Format {format_ext} not available: {e}")

    def test_opencv_combine_video_image_format(self):
        """Test OpenCV video combining with image formats (should fall back to Pillow)."""
        output_path = os.path.join(self.temp_dir, "test_opencv.gif")

        result_path = opencv_combine_video(
            image_batch=self.test_frames,
            output_path=output_path,
            frame_rate=10,
            video_format="image/gif"
        )

        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_both_implementations_same_input(self):
        """Test that both implementations handle the same input correctly."""
        ffmpeg_output = os.path.join(self.temp_dir, "test_ffmpeg_comparison.mp4")
        opencv_output = os.path.join(self.temp_dir, "test_opencv_comparison.mp4")

        # Use same parameters for both
        ffmpeg_result = ffmpeg_combine_video(
            image_batch=self.test_frames,
            output_path=ffmpeg_output,
            frame_rate=10,
            video_format="video/h264-mp4"
        )

        opencv_result = opencv_combine_video(
            image_batch=self.test_frames,
            output_path=opencv_output,
            frame_rate=10,
            video_format="video/mp4"
        )

        # Both should succeed
        self.assertEqual(ffmpeg_result, ffmpeg_output)
        self.assertEqual(opencv_result, opencv_output)
        self.assertTrue(os.path.exists(ffmpeg_output))
        self.assertTrue(os.path.exists(opencv_output))

    def test_empty_frame_list(self):
        """Test video combining with empty frame list."""
        output_path = os.path.join(self.temp_dir, "test_empty.mp4")

        # Test with empty frame list
        with self.assertRaises(Exception):
            ffmpeg_combine_video(
                image_batch=[],
                output_path=output_path,
                frame_rate=10,
                video_format="video/h264-mp4"
            )

        with self.assertRaises(Exception):
            opencv_combine_video(
                image_batch=[],
                output_path=output_path,
                frame_rate=10,
                video_format="video/mp4"
            )

    def test_single_frame_video(self):
        """Test video combining with single frame."""
        output_path = os.path.join(self.temp_dir, "test_single.mp4")
        single_frame = [Image.new("RGB", (100, 100), color=(255, 0, 0))]

        result_path = ffmpeg_combine_video(
            image_batch=single_frame,
            output_path=output_path,
            frame_rate=10,
            video_format="video/h264-mp4"
        )

        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_different_frame_sizes(self):
        """Test video combining with different frame sizes."""
        output_path = os.path.join(self.temp_dir, "test_varying_sizes.mp4")

        # Create frames with different sizes
        varying_frames = [
            Image.new("RGB", (100, 100), color=(255, 0, 0)),
            Image.new("RGB", (120, 80), color=(0, 255, 0)),
            Image.new("RGB", (80, 120), color=(0, 0, 255))
        ]

        # Both implementations should handle varying frame sizes
        for combine_func in [ffmpeg_combine_video, opencv_combine_video]:
            with self.subTest(func=combine_func.__name__):
                result_path = combine_func(
                    image_batch=varying_frames,
                    output_path=output_path,
                    frame_rate=10,
                    video_format="video/mp4" if combine_func == opencv_combine_video else "video/h264-mp4"
                )

                self.assertEqual(result_path, output_path)
                self.assertTrue(os.path.exists(output_path))
                self.assertGreater(os.path.getsize(output_path), 0)


if __name__ == "__main__":
    unittest.main()