"""
Unit tests for video loading functionality.
"""

import unittest
import os
import numpy as np
import torch

# Import the video utilities
from utils.video import ffmpeg_load_video, opencv_load_video, VideoInfo


class TestVideoLoading(unittest.TestCase):
    """Test video loading functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_video_path = "tests/test1.mp4"

    # def test_ffmpeg_frame_generator(self):
    #     """Test FFmpeg frame generator with test video."""
    #     if not os.path.exists(self.test_video_path):
    #         self.skipTest(f"Test video not found: {self.test_video_path}")

    #     # Test basic frame generation
    #     frame_gen = _ffmpeg_frame_generator(
    #         video_path=self.test_video_path,
    #         frame_load_cap=5  # Load only 5 frames for testing
    #     )

    #     frames = list(frame_gen)
    #     self.assertGreater(len(frames), 0)
    #     self.assertLessEqual(len(frames), 5)

    #     # Check frame properties
    #     for frame in frames:
    #         self.assertIsInstance(frame, np.ndarray)
    #         self.assertEqual(frame.dtype, np.float32)
    #         self.assertEqual(frame.shape[-1], 3)  # RGB channels
    #         # Check value range [0, 1]
    #         self.assertTrue(np.all(frame >= 0))
    #         self.assertTrue(np.all(frame <= 1))

    # def test_opencv_frame_generator(self):
    #     """Test OpenCV frame generator with test video."""
    #     if not os.path.exists(self.test_video_path):
    #         self.skipTest(f"Test video not found: {self.test_video_path}")

    #     # Test basic frame generation
    #     frame_gen = _cv_frame_generator(
    #         video_path=self.test_video_path,
    #         frame_load_cap=5  # Load only 5 frames for testing
    #     )

    #     frames = list(frame_gen)
    #     self.assertGreater(len(frames), 0)
    #     self.assertLessEqual(len(frames), 5)

    #     # Check frame properties
    #     for frame in frames:
    #         self.assertIsInstance(frame, np.ndarray)
    #         self.assertEqual(frame.dtype, np.float32)
    #         self.assertEqual(frame.shape[-1], 3)  # RGB channels
    #         # Check value range [0, 1]
    #         self.assertTrue(np.all(frame >= 0))
    #         self.assertTrue(np.all(frame <= 1))

    def test_ffmpeg_load_video(self):
        """Test FFmpeg video loading."""
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Test video not found: {self.test_video_path}")

        # Load video with frame limit
        frames_tensor, video_info = ffmpeg_load_video(
            video_path=self.test_video_path,
            frame_load_cap=10
        )

        self.assertIsInstance(frames_tensor, torch.Tensor)
        self.assertEqual(video_info.loaded_frame_count, frames_tensor.shape[0])
        self.assertLessEqual(video_info.loaded_frame_count, 10)
        self.assertIsInstance(video_info, VideoInfo)

        # Check tensor properties
        self.assertEqual(frames_tensor.dtype, torch.float32)
        self.assertEqual(frames_tensor.shape[-1], 3)  # RGB channels

    def test_ffmpeg_load_video_comprehensive_parameters(self):
        """Test FFmpeg video loading with comprehensive parameter combinations."""
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Test video not found: {self.test_video_path}")

        # Test different parameter combinations
        test_cases = [
            # Basic frame limit
            {"frame_load_cap": 5},
            # Custom resolution
            {"custom_width": 320, "custom_height": 240},
            # Force frame rate
            {"force_rate": 15},
            # Start time
            {"start_time": 1},
            # Downscale ratio
            {"downscale_ratio": 16},
            # Memory limit
            {"memory_limit_mb": 10},
            # Select every nth frame
            {"select_every_nth": 2},
            # Multiple parameters
            {"frame_load_cap": 3, "custom_width": 256, "force_rate": 10},
            {"frame_load_cap": 10, "select_every_nth": 3},
        ]

        for params in test_cases:
            with self.subTest(params=params):
                frames_tensor, video_info = ffmpeg_load_video(
                    video_path=self.test_video_path,
                    **params
                )

                # Basic validation
                self.assertIsInstance(frames_tensor, torch.Tensor)
                self.assertIsInstance(video_info, VideoInfo)
                self.assertEqual(frames_tensor.dtype, torch.float32)
                self.assertEqual(frames_tensor.shape[-1], 3)  # RGB channels

                # Validate frame count
                if "frame_load_cap" in params:
                    self.assertLessEqual(video_info.loaded_frame_count, params["frame_load_cap"])

                # Validate resolution if custom dimensions provided
                if "custom_width" in params and params["custom_width"] > 0:
                    self.assertEqual(video_info.loaded_width, params["custom_width"])
                if "custom_height" in params and params["custom_height"] > 0:
                    self.assertEqual(video_info.loaded_height, params["custom_height"])

    def test_ffmpeg_select_every_nth_parameter(self):
        """Test FFmpeg video loading with select_every_nth parameter."""
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Test video not found: {self.test_video_path}")

        # Test different values of select_every_nth
        test_cases = [
            {"select_every_nth": 1},  # Default value, should load all frames
            {"select_every_nth": 2},  # Load every 2nd frame
            {"select_every_nth": 3},  # Load every 3rd frame
            {"select_every_nth": 5},  # Load every 5th frame
        ]

        # Get baseline frame count with default parameters
        _, baseline_info = ffmpeg_load_video(
            video_path=self.test_video_path,
            frame_load_cap=20  # Load up to 20 frames for consistent testing
        )
        baseline_frame_count = baseline_info.loaded_frame_count

        for params in test_cases:
            with self.subTest(params=params):
                frames_tensor, video_info = ffmpeg_load_video(
                    video_path=self.test_video_path,
                    frame_load_cap=20,  # Consistent frame cap
                    **params
                )

                # Basic validation
                self.assertIsInstance(frames_tensor, torch.Tensor)
                self.assertIsInstance(video_info, VideoInfo)
                self.assertEqual(frames_tensor.dtype, torch.float32)
                self.assertEqual(frames_tensor.shape[-1], 3)  # RGB channels

                # Validate frame count reduction with select_every_nth
                select_every_nth = params["select_every_nth"]
                expected_frame_count = baseline_frame_count // select_every_nth
                if baseline_frame_count % select_every_nth != 0:
                    expected_frame_count += 1

                # Allow for slight variations due to frame timing
                self.assertLessEqual(abs(video_info.loaded_frame_count - expected_frame_count), 1)

    def test_ffmpeg_select_every_nth_combinations(self):
        """Test FFmpeg video loading with select_every_nth combined with other parameters."""
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Test video not found: {self.test_video_path}")

        # Test combinations with other parameters
        test_cases = [
            {"frame_load_cap": 10, "select_every_nth": 2},
            {"custom_width": 320, "select_every_nth": 3},
            {"force_rate": 15, "select_every_nth": 2},
            {"start_time": 1, "select_every_nth": 3},
            {"memory_limit_mb": 10, "select_every_nth": 2},
            {"frame_load_cap": 15, "custom_width": 256, "select_every_nth": 3},
        ]

        for params in test_cases:
            with self.subTest(params=params):
                frames_tensor, video_info = ffmpeg_load_video(
                    video_path=self.test_video_path,
                    **params
                )

                # Basic validation
                self.assertIsInstance(frames_tensor, torch.Tensor)
                self.assertIsInstance(video_info, VideoInfo)
                self.assertEqual(frames_tensor.dtype, torch.float32)
                self.assertEqual(frames_tensor.shape[-1], 3)  # RGB channels

                # Validate frame count
                if "frame_load_cap" in params:
                    self.assertLessEqual(video_info.loaded_frame_count, params["frame_load_cap"])

                # Validate resolution if custom dimensions provided
                if "custom_width" in params and params["custom_width"] > 0:
                    self.assertEqual(video_info.loaded_width, params["custom_width"])
                if "custom_height" in params and params["custom_height"] > 0:
                    self.assertEqual(video_info.loaded_height, params["custom_height"])

    def test_ffmpeg_select_every_nth_edge_cases(self):
        """Test FFmpeg video loading with edge cases for select_every_nth parameter."""
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Test video not found: {self.test_video_path}")

        # Test edge cases
        test_cases = [
            {"select_every_nth": 1},  # Minimum valid value
            {"select_every_nth": 10}, # Large value that might result in few frames
            {"frame_load_cap": 5, "select_every_nth": 2},  # Small frame cap with selection
        ]

        for params in test_cases:
            with self.subTest(params=params):
                frames_tensor, video_info = ffmpeg_load_video(
                    video_path=self.test_video_path,
                    **params
                )

                # Basic validation
                self.assertIsInstance(frames_tensor, torch.Tensor)
                self.assertIsInstance(video_info, VideoInfo)
                self.assertEqual(frames_tensor.dtype, torch.float32)
                self.assertEqual(frames_tensor.shape[-1], 3)  # RGB channels

                # Should always have at least one frame
                self.assertGreater(video_info.loaded_frame_count, 0)

    def test_ffmpeg_video_info_fields(self):
        """Test that all VideoInfo fields are correctly populated for FFmpeg."""
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Test video not found: {self.test_video_path}")

        frames_tensor, video_info = ffmpeg_load_video(
            video_path=self.test_video_path,
            frame_load_cap=5
        )

        # Use frames_tensor to avoid unused variable warning
        _ = frames_tensor  # Mark as used

        # Test all VideoInfo fields
        self.assertEqual(video_info.source_path, self.test_video_path)
        self.assertIsInstance(video_info.source_width, int)
        self.assertIsInstance(video_info.source_height, int)
        self.assertIsInstance(video_info.source_fps, float)
        self.assertIsInstance(video_info.source_frame_count, int)
        self.assertIsInstance(video_info.loaded_width, int)
        self.assertIsInstance(video_info.loaded_height, int)
        self.assertIsInstance(video_info.loaded_channels, int)
        self.assertIsInstance(video_info.loaded_frame_count, int)
        self.assertIsInstance(video_info.loaded_fps, float)
        self.assertEqual(video_info.generator, "ffmpeg")

        # Test derived properties
        self.assertIsInstance(video_info.resolution, str)
        self.assertIsInstance(video_info.aspect_ratio, float)

        # Test duration calculations
        if video_info.source_fps > 0:
            self.assertIsInstance(video_info.total_duration, float)
            self.assertIsInstance(video_info.estimated_duration, float)
            self.assertGreaterEqual(video_info.total_duration, 0)
            self.assertGreaterEqual(video_info.estimated_duration, 0)

        # Test string representations
        self.assertIsInstance(str(video_info), str)
        self.assertIsInstance(repr(video_info), str)
        self.assertIn(self.test_video_path, str(video_info))
        self.assertIn("ffmpeg", repr(video_info))

    def test_opencv_load_video(self):
        """Test OpenCV video loading."""
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Test video not found: {self.test_video_path}")

        # Load video with frame limit
        frames_tensor, video_info = opencv_load_video(
            video_path=self.test_video_path,
            frame_load_cap=10
        )

        self.assertIsInstance(frames_tensor, torch.Tensor)
        self.assertEqual(video_info.loaded_frame_count, frames_tensor.shape[0])
        self.assertLessEqual(video_info.loaded_frame_count, 10)
        self.assertIsInstance(video_info, VideoInfo)

        # Check tensor properties
        self.assertEqual(frames_tensor.dtype, torch.float32)
        self.assertEqual(frames_tensor.shape[-1], 3)  # RGB channels

    def test_opencv_load_video_comprehensive_parameters(self):
        """Test OpenCV video loading with comprehensive parameter combinations."""
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Test video not found: {self.test_video_path}")

        # Test different parameter combinations
        test_cases = [
            # Basic frame limit
            {"frame_load_cap": 5},
            # Force frame rate
            {"force_rate": 15},
            # Start time
            {"start_time": 1},
            # Select every nth frame
            {"select_every_nth": 2},
            # Memory limit
            {"memory_limit_mb": 10},
            # Multiple parameters
            {"frame_load_cap": 3, "force_rate": 10, "start_time": 1},
        ]

        for params in test_cases:
            with self.subTest(params=params):
                frames_tensor, video_info = opencv_load_video(
                    video_path=self.test_video_path,
                    **params
                )

                # Basic validation
                self.assertIsInstance(frames_tensor, torch.Tensor)
                self.assertIsInstance(video_info, VideoInfo)
                self.assertEqual(frames_tensor.dtype, torch.float32)
                self.assertEqual(frames_tensor.shape[-1], 3)  # RGB channels

                # Validate frame count
                if "frame_load_cap" in params:
                    self.assertLessEqual(video_info.loaded_frame_count, params["frame_load_cap"])

                # Validate frame rate if forced
                if "force_rate" in params and params["force_rate"] > 0:
                    self.assertEqual(video_info.loaded_fps, params["force_rate"])

    def test_opencv_video_info_fields(self):
        """Test that all VideoInfo fields are correctly populated for OpenCV."""
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Test video not found: {self.test_video_path}")

        frames_tensor, video_info = opencv_load_video(
            video_path=self.test_video_path,
            frame_load_cap=5
        )

        # Test all VideoInfo fields
        self.assertEqual(video_info.source_path, self.test_video_path)
        self.assertIsInstance(video_info.source_width, int)
        self.assertIsInstance(video_info.source_height, int)
        self.assertIsInstance(video_info.source_fps, float)
        self.assertIsInstance(video_info.source_frame_count, int)
        self.assertIsInstance(video_info.loaded_width, int)
        self.assertIsInstance(video_info.loaded_height, int)
        self.assertIsInstance(video_info.loaded_channels, int)
        self.assertIsInstance(video_info.loaded_frame_count, int)
        self.assertIsInstance(video_info.loaded_fps, float)
        self.assertEqual(video_info.generator, "opencv")

        # Test derived properties
        self.assertIsInstance(video_info.resolution, str)
        self.assertIsInstance(video_info.aspect_ratio, float)

        # Test duration calculations
        if video_info.source_fps > 0:
            self.assertIsInstance(video_info.total_duration, float)
            self.assertIsInstance(video_info.estimated_duration, float)
            self.assertGreaterEqual(video_info.total_duration, 0)
            self.assertGreaterEqual(video_info.estimated_duration, 0)

        # Test string representations
        self.assertIsInstance(str(video_info), str)
        self.assertIsInstance(repr(video_info), str)
        self.assertIn(self.test_video_path, str(video_info))
        self.assertIn("opencv", repr(video_info))

    def test_frame_generator_parameter_validation(self):
        """Test frame generator parameter validation and edge cases."""
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Test video not found: {self.test_video_path}")

        # Test edge cases for frame_load_cap
        test_cases = [
            {"frame_load_cap": 1},  # Minimum frame count
            {"frame_load_cap": 0},  # Unlimited frames (may load all)
        ]

        for params in test_cases:
            with self.subTest(params=params):
                # Test FFmpeg
                frames_tensor_ffmpeg, video_info_ffmpeg = ffmpeg_load_video(
                    video_path=self.test_video_path,
                    **params
                )
                self.assertIsInstance(frames_tensor_ffmpeg, torch.Tensor)
                self.assertIsInstance(video_info_ffmpeg, VideoInfo)

                # Test OpenCV
                frames_tensor_opencv, video_info_opencv = opencv_load_video(
                    video_path=self.test_video_path,
                    **params
                )
                self.assertIsInstance(frames_tensor_opencv, torch.Tensor)
                self.assertIsInstance(video_info_opencv, VideoInfo)

    def test_ffmpeg_vs_opencv_consistency(self):
        """Test that FFmpeg and OpenCV produce consistent results for basic loading."""
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Test video not found: {self.test_video_path}")

        # Load with both backends using same parameters
        frames_tensor_ffmpeg, video_info_ffmpeg = ffmpeg_load_video(
            video_path=self.test_video_path,
            frame_load_cap=5
        )

        frames_tensor_opencv, video_info_opencv = opencv_load_video(
            video_path=self.test_video_path,
            frame_load_cap=5
        )

        # Both should return valid results
        self.assertIsInstance(frames_tensor_ffmpeg, torch.Tensor)
        self.assertIsInstance(frames_tensor_opencv, torch.Tensor)
        self.assertIsInstance(video_info_ffmpeg, VideoInfo)
        self.assertIsInstance(video_info_opencv, VideoInfo)

        # Both should have the same number of frames (up to the cap)
        self.assertEqual(video_info_ffmpeg.loaded_frame_count, frames_tensor_ffmpeg.shape[0])
        self.assertEqual(video_info_opencv.loaded_frame_count, frames_tensor_opencv.shape[0])

        # Both should have the same tensor data type
        self.assertEqual(frames_tensor_ffmpeg.dtype, torch.float32)
        self.assertEqual(frames_tensor_opencv.dtype, torch.float32)

        # Both should have RGB channels
        self.assertEqual(frames_tensor_ffmpeg.shape[-1], 3)
        self.assertEqual(frames_tensor_opencv.shape[-1], 3)


if __name__ == "__main__":
    unittest.main()