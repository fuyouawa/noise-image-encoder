"""
主应用类模块
包含应用程序的主要逻辑和界面
"""
import tkinter as tk
from tkinter import messagebox

from gui_components import DragDropBox
from config_manager import ConfigManager
from resource_processor import ResourceProcessor


class SteganographyEncoder:
    """隐写图像编码器主应用类"""

    def __init__(self, root):
        self.root = root
        self.root.title("隐写图像编码器")
        self.root.geometry("700x650")

        # Initialize managers
        self.config_manager = ConfigManager(on_always_on_top_changed=self._on_always_on_top_changed)
        self.resource_processor = ResourceProcessor()

        # Create configuration frame
        self.config_manager.create_config_frame(root)

        # Create main frame
        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Create drag and drop boxes
        self.create_drag_drop_boxes(main_frame)

        # Status label
        self.status_label = tk.Label(root, text="拖拽资源文件到上方方框", font=("Arial", 10))
        self.status_label.pack(side="bottom", pady=10)

    def create_drag_drop_boxes(self, parent):
        """创建拖拽框"""
        # Left box for regular images -> noise images
        left_frame = tk.Frame(parent)
        left_frame.pack(side="left", fill="both", expand=True, padx=10)

        self.left_box = DragDropBox(left_frame, "拖拽资源文件到这里\n(转换为隐写图像)", self.process_resource_to_steganography)
        self.left_box.frame.pack(fill="both", expand=True)

        # Right box for steganography images -> regular images
        right_frame = tk.Frame(parent)
        right_frame.pack(side="right", fill="both", expand=True, padx=10)

        self.right_box = DragDropBox(right_frame, "拖拽隐写图像文件到这里\n(转换为原始资源文件)", self.process_steganography_to_resource)
        self.right_box.frame.pack(fill="both", expand=True)

    def process_resource_to_steganography(self, file_path):
        """处理资源文件转换为隐写图像"""
        try:
            self.status_label.config(text="正在处理资源文件...")

            # Get configuration
            compression_level = self.config_manager.get_compression_level()
            steganography_width = self.config_manager.get_steganography_width()
            steganography_height = self.config_manager.get_steganography_height()
            use_alpha = self.config_manager.get_use_alpha()
            top_margin_ratio = self.config_manager.get_encode_top_margin_ratio()
            bottom_margin_ratio = self.config_manager.get_encode_bottom_margin_ratio()

            # Process resource file
            steganography_image = self.resource_processor.process_resource_to_steganography(
                file_path, compression_level, steganography_width, steganography_height, use_alpha, top_margin_ratio, bottom_margin_ratio
            )

            # Save steganography image
            output_path = self.resource_processor.get_output_path(file_path, "_steganography", custom_extension="png")
            self.resource_processor.save_steganography_image(steganography_image, output_path)

            self.status_label.config(text=f"转换成功！隐写图像已保存到: {output_path}")
            messagebox.showinfo("成功", f"隐写图像已保存到:\n{output_path}")

        except Exception as e:
            self.status_label.config(text="处理失败")
            messagebox.showerror("错误", f"处理资源文件时出错:\n{str(e)}")

    def process_steganography_to_resource(self, file_path):
        """处理隐写图像转换为原始资源文件"""
        try:
            self.status_label.config(text="正在处理隐写图像...")

            # Get configuration
            top_margin_ratio = self.config_manager.get_decode_top_margin_ratio()
            bottom_margin_ratio = self.config_manager.get_decode_bottom_margin_ratio()

            # Process steganography image
            file_bytes, file_extension = self.resource_processor.process_steganography_to_resource(file_path, top_margin_ratio, bottom_margin_ratio)

            # Save resource file
            output_path = self.resource_processor.get_output_path(file_path, "_decoded", custom_extension=file_extension)
            with open(output_path, 'wb') as f:
                f.write(file_bytes)

            self.status_label.config(text=f"转换成功！资源文件已保存到: {output_path}")
            messagebox.showinfo("成功", f"资源文件已保存到:\n{output_path}")

        except Exception as e:
            self.status_label.config(text="处理失败")
            messagebox.showerror("错误", f"处理隐写图像时出错:\n{str(e)}")

    def _on_always_on_top_changed(self, always_on_top):
        """置顶状态改变时的回调"""
        self.root.attributes("-topmost", always_on_top)
