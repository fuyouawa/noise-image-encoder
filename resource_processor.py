"""
资源文件处理模块
包含文件编码、解码、隐写图像转换等处理逻辑
"""
import zlib
import os
from PIL import Image
import numpy as np
import torch

from utils.encoding import encode_steganography, decode_steganography
from utils.format import file_extension_to_mime_type, mime_type_to_file_extension
from utils.serialization import ResourceHeader, CompressionMode


class ResourceProcessor:
    """资源文件处理器"""

    def __init__(self):
        pass

    def process_resource_to_steganography(self, file_path, compression_level, steganography_width, steganography_height, use_alpha):
        """处理普通资源文件转换为隐写图像"""
        try:
            # Read file as raw bytes
            with open(file_path, 'rb') as f:
                file_bytes = f.read()

            # Compress with zlib using configured compression level
            compressed_data = zlib.compress(file_bytes, compression_level)

            # Get file extension and convert to MIME type
            file_extension = os.path.splitext(file_path)[1].lower().lstrip('.')
            mime_type = file_extension_to_mime_type(file_extension)

            # Create ResourceHeader with MIME type and zlib compression
            resource_header = ResourceHeader.from_mime_type(
                mime_type=mime_type,
                compression_mode=CompressionMode.ZLIB_COMPRESSION
            )

            # Combine header and compressed data
            header_bytes = resource_header.to_bytes()
            data_with_header = header_bytes + compressed_data

            # Convert to steganography image with configured dimensions
            width = steganography_width if steganography_width > 0 else None
            height = steganography_height if steganography_height > 0 else None

            steganography_image = encode_steganography(data_with_header, width=width, height=height, use_alpha=use_alpha)

            return steganography_image

        except Exception as e:
            raise Exception(f"处理文件时出错: {str(e)}")

    def process_steganography_to_resource(self, file_path):
        """处理隐写图像转换为原始资源文件"""
        try:
            # Load steganography image
            steganography_image = self._load_steganography_image(file_path)

            # Decode bytes from steganography image
            data_with_header = decode_steganography(steganography_image)

            # Extract ResourceHeader (first 4 bytes)
            header_bytes = data_with_header[:4]
            compressed_data = data_with_header[4:]

            # Parse ResourceHeader
            resource_header = ResourceHeader.from_bytes(header_bytes)

            # Get file extension from resource header mime_type using mime_type_to_file_extension
            file_extension = mime_type_to_file_extension(resource_header.mime_type)

            # Decompress with zlib if compression is enabled
            if resource_header.compression_mode == CompressionMode.ZLIB_COMPRESSION:
                file_bytes = zlib.decompress(compressed_data)
            else:
                file_bytes = compressed_data

            return file_bytes, file_extension

        except Exception as e:
            raise Exception(f"处理隐写图像时出错: {str(e)}")

    def _load_steganography_image(self, file_path):
        """加载隐写图像并转换为张量"""
        with Image.open(file_path) as img:
            img_array = np.array(img)

        # Convert to float32 [0, 1] range
        if img_array.dtype == np.uint8:
            img_array = img_array.astype(np.float32) / 255.0

        # Add batch dimension
        image_tensor = torch.from_numpy(img_array).unsqueeze(0)

        return image_tensor

    def save_steganography_image(self, steganography_image, output_path):
        """保存隐写图像"""
        # Convert tensor to PIL image and save
        image_array = steganography_image[0].detach().cpu().numpy()
        image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)

        # Determine image mode based on number of channels
        if image_array.shape[2] == 4:
            pil_image = Image.fromarray(image_array, 'RGBA')
        else:
            pil_image = Image.fromarray(image_array, 'RGB')

        pil_image.save(output_path)

    def save_regular_image(self, image_tensor, output_path):
        """保存普通图像"""
        # Convert tensor to PIL image and save
        image_array = image_tensor[0].detach().cpu().numpy()
        image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)

        pil_image = Image.fromarray(image_array)
        pil_image.save(output_path)

    def get_output_path(self, input_path, suffix, custom_extension=None):
        """生成输出文件路径"""
        directory = os.path.dirname(input_path)
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)

        if custom_extension is not None:
            ext = f".{custom_extension}"
        return os.path.join(directory, f"{name}{suffix}{ext}")
