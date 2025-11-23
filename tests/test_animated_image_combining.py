"""
Unit tests for video combining functionality.
"""

import unittest
import tempfile
import os
from PIL import Image

# Import the video utilities
from utils.video.common import combine_animated_image


class TestAnimatedImageCombining(unittest.TestCase):
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

    def test_combine_animated_image_gif(self):
        """Test animated GIF creation."""
        output_path = os.path.join(self.temp_dir, "test.gif")

        result_path = combine_animated_image(
            frames=self.test_frames,
            output_path=output_path,
            format_ext="gif",
            frame_rate=10,
            loop_count=0
        )

        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))

        # Verify the created file
        with Image.open(output_path) as img:
            self.assertEqual(img.format, "GIF")

    def test_combine_animated_image_webp(self):
        """Test animated WebP creation."""
        output_path = os.path.join(self.temp_dir, "test.webp")

        result_path = combine_animated_image(
            frames=self.test_frames,
            output_path=output_path,
            format_ext="webp",
            frame_rate=10,
            loop_count=0
        )

        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))


if __name__ == "__main__":
    unittest.main()