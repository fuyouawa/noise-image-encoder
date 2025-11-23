"""
Unit tests for common video utilities.
"""

import unittest
import numpy as np
import torch
from PIL import Image

# Import the video utilities
from utils.video.common import (
    image_batch_to_pil_list,
    combine_animated_image,
    target_size,
    batched,
    frames_to_tensor,
    _ensure_even_dimensions
)


class TestCommonUtilities(unittest.TestCase):
    """Test common video utilities."""

    def test_convert_image_batch_to_pil_list(self):
        """Test image batch conversion to PIL list."""
        # Test with numpy arrays
        numpy_images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)]
        pil_list = image_batch_to_pil_list(numpy_images)

        self.assertEqual(len(pil_list), 3)
        for img in pil_list:
            self.assertIsInstance(img, Image.Image)
            self.assertEqual(img.mode, "RGB")

    def test_convert_image_batch_with_pil_images(self):
        """Test image batch conversion with PIL images."""
        pil_images = [Image.new("RGB", (100, 100), color=(255, 0, 0)) for _ in range(2)]
        converted = image_batch_to_pil_list(pil_images)

        self.assertEqual(len(converted), 2)
        for img in converted:
            self.assertIsInstance(img, Image.Image)
            self.assertEqual(img.mode, "RGB")

    def test_ensure_even_dimensions(self):
        """Test even dimension enforcement."""
        # Test odd dimensions
        img_odd = Image.new("RGB", (101, 101))
        img_even = _ensure_even_dimensions(img_odd)
        self.assertEqual(img_even.size, (102, 102))

        # Test even dimensions (should remain unchanged)
        img_even_orig = Image.new("RGB", (100, 100))
        img_even_result = _ensure_even_dimensions(img_even_orig)
        self.assertEqual(img_even_result.size, (100, 100))

    def test_target_size_calculation(self):
        """Test target size calculation with different parameters."""
        # Test auto calculation
        width, height = target_size(720, 1280, 0, 0)
        self.assertEqual(width, 720)
        self.assertEqual(height, 1280)

        # Test custom width
        width, height = target_size(720, 1280, 360, 0)
        self.assertEqual(width, 360)
        self.assertEqual(height, 640)

        # Test custom height
        width, height = target_size(720, 1280, 0, 640)
        self.assertEqual(width, 360)
        self.assertEqual(height, 640)

        # Test both custom
        width, height = target_size(720, 1280, 360, 640)
        self.assertEqual(width, 360)
        self.assertEqual(height, 640)

    def test_batched_generator(self):
        """Test batch generator utility."""
        data = list(range(10))
        batches = list(batched(iter(data), 3))

        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[0], (0, 1, 2))
        self.assertEqual(batches[1], (3, 4, 5))
        self.assertEqual(batches[2], (6, 7, 8))
        self.assertEqual(batches[3], (9,))

    def test_frames_to_tensor(self):
        """Test frame generator to tensor conversion."""
        # Create dummy frame generator
        def dummy_frame_generator():
            for i in range(5):
                yield np.random.rand(64, 64, 3).astype(np.float32)

        tensor = frames_to_tensor(dummy_frame_generator(), 64, 64, 3)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (5, 64, 64, 3))
        self.assertEqual(tensor.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()