"""
Integration tests for video utilities.
"""

import unittest
import tempfile
import os
import numpy as np
import torch
from PIL import Image

# Import the video utilities
from utils.video.common import VideoInfo, combine_animated_image
from utils.video.ffmpeg import load_video as ffmpeg_load_video


class TestIntegration(unittest.TestCase):
    """Integration tests for video utilities."""

    def setUp(self):
        """Set up test environment."""
        self.test_video_path = "tests/test1.mp4"
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_video_round_trip(self):
        """Test loading and then combining video frames."""
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Test video not found: {self.test_video_path}")

        # Load video frames
        frames_tensor, video_info = ffmpeg_load_video(
            video_path=self.test_video_path,
            frame_load_cap=5
        )

        # Convert tensor back to PIL images
        frames_list = []
        for i in range(min(3, video_info.loaded_frame_count)):  # Use first 3 frames
            frame_np = frames_tensor[i].numpy()
            frame_np = (frame_np * 255).astype(np.uint8)
            img = Image.fromarray(frame_np)
            frames_list.append(img)

        # Combine frames back to video
        output_path = os.path.join(self.temp_dir, "output.gif")
        result_path = combine_animated_image(
            frames=frames_list,
            output_path=output_path,
            format_ext="gif",
            frame_rate=10,
            loop_count=0
        )

        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))

    def test_memory_limit_handling(self):
        """Test memory limit calculation and handling."""
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Test video not found: {self.test_video_path}")

        # Test with very low memory limit
        frames_tensor, video_info = ffmpeg_load_video(
            video_path=self.test_video_path,
            memory_limit_mb=1  # 1MB limit
        )

        self.assertIsInstance(frames_tensor, torch.Tensor)
        self.assertGreater(video_info.loaded_frame_count, 0)
        # With 1MB limit, should load very few frames
        self.assertLess(video_info.loaded_frame_count, 10)


if __name__ == "__main__":
    unittest.main()