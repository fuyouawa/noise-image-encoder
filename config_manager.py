"""
配置管理模块
管理应用程序的配置选项
"""
import tkinter as tk
from accordion_widget import AccordionWidget


class ConfigManager:
    """配置管理器"""

    def __init__(self, on_always_on_top_changed=None):
        # Configuration variables
        self.compression_level = tk.IntVar(value=-1)
        self.steganography_width = tk.IntVar(value=0)
        self.steganography_height = tk.IntVar(value=0)
        self.use_alpha = tk.BooleanVar(value=False)
        self.top_margin_ratio = tk.DoubleVar(value=20.0)  # 20% top margin (stored as percentage)
        self.bottom_margin_ratio = tk.DoubleVar(value=20.0)  # 20% bottom margin (stored as percentage)
        self.always_on_top = tk.BooleanVar(value=False)  # 窗口置顶选项
        self.on_always_on_top_changed = on_always_on_top_changed  # 置顶状态改变回调

        # Video synthesis configuration variables
        self.video_synthesis_mode = tk.StringVar(value="ffmpeg")  # ffmpeg or opencv
        self.video_frame_rate = tk.IntVar(value=16)  # Video frame rate
        self.image_encryption_method = tk.StringVar(value="invert")  # none, invert, xor-16, xor-32, xor-64, xor-128

    def create_config_frame(self, parent):
        """创建配置选项框架"""
        # 创建手风琴控件
        accordion = AccordionWidget(parent)

        # 通用选项分组
        general_content = tk.Frame(accordion.main_frame)

        # Compression level
        tk.Label(general_content, text="压缩等级:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        compression_frame = tk.Frame(general_content)
        compression_frame.grid(row=0, column=1, sticky="w")

        compression_scale = tk.Scale(compression_frame, from_=-1, to=9, orient="horizontal",
                                   variable=self.compression_level, showvalue=True, length=200)
        compression_scale.pack(side="left")
        tk.Label(compression_frame, text="(-1=默认, 0=无压缩, 9=最大压缩)", font=("Arial", 8)).pack(side="left", padx=(5, 0))

        # Margin ratios
        tk.Label(general_content, text="上预留区域:").grid(row=3, column=0, sticky="w", padx=(0, 5), pady=(10, 0))
        top_margin_frame = tk.Frame(general_content)
        top_margin_frame.grid(row=3, column=1, sticky="w", pady=(10, 0))

        top_margin_scale = tk.Scale(top_margin_frame, from_=0, to=100, orient="horizontal",
                                   variable=self.top_margin_ratio, showvalue=True, length=150, resolution=1)
        top_margin_scale.pack(side="left")
        tk.Label(top_margin_frame, text="%", font=("Arial", 8)).pack(side="left", padx=(5, 0))

        tk.Label(general_content, text="下预留区域:").grid(row=4, column=0, sticky="w", padx=(0, 5), pady=(10, 0))
        bottom_margin_frame = tk.Frame(general_content)
        bottom_margin_frame.grid(row=4, column=1, sticky="w", pady=(10, 0))

        bottom_margin_scale = tk.Scale(bottom_margin_frame, from_=0, to=100, orient="horizontal",
                                      variable=self.bottom_margin_ratio, showvalue=True, length=150, resolution=1)
        bottom_margin_scale.pack(side="left")
        tk.Label(bottom_margin_frame, text="%", font=("Arial", 8)).pack(side="left", padx=(5, 0))

        # Image decryption method
        tk.Label(general_content, text="图像加密/解密:").grid(row=5, column=0, sticky="w", padx=(0, 5), pady=(10, 0))
        encryption_frame = tk.Frame(general_content)
        encryption_frame.grid(row=5, column=1, sticky="w", pady=(10, 0))

        encryption_options = [
            ("无", "none"),
            ("颜色反转", "invert"),
            ("异或-密匙16", "xor-16"),
            ("异或-密匙32", "xor-32"),
            ("异或-密匙64", "xor-64"),
            ("异或-密匙128", "xor-128")
        ]

        for i, (label, value) in enumerate(encryption_options):
            radio = tk.Radiobutton(encryption_frame, text=label, variable=self.image_encryption_method, value=value)
            radio.grid(row=0, column=i, sticky="w", padx=(0, 10))

        # 编码选项分组
        encoding_content = tk.Frame(accordion.main_frame)

        # Steganography image dimensions
        tk.Label(encoding_content, text="隐写图像尺寸:").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(10, 0))
        dimensions_frame = tk.Frame(encoding_content)
        dimensions_frame.grid(row=1, column=1, sticky="w", pady=(10, 0))

        tk.Label(dimensions_frame, text="宽:").pack(side="left")
        width_entry = tk.Entry(dimensions_frame, textvariable=self.steganography_width, width=6)
        width_entry.pack(side="left", padx=(2, 10))

        tk.Label(dimensions_frame, text="高:").pack(side="left")
        height_entry = tk.Entry(dimensions_frame, textvariable=self.steganography_height, width=6)
        height_entry.pack(side="left", padx=(2, 0))

        tk.Label(dimensions_frame, text="(0=不指定)", font=("Arial", 8)).pack(side="left", padx=(5, 0))

        # Alpha channel option
        tk.Label(encoding_content, text="使用alpha通道:").grid(row=2, column=0, sticky="w", padx=(0, 5), pady=(10, 0))
        alpha_check = tk.Checkbutton(encoding_content, variable=self.use_alpha)
        alpha_check.grid(row=2, column=1, sticky="w", pady=(10, 0))

        # 解码选项分组
        decoding_content = tk.Frame(accordion.main_frame)

        # 图像批次合成视频标题
        tk.Label(decoding_content, text="图像批次合成视频", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="s", pady=(0, 0))

        # Video synthesis mode
        tk.Label(decoding_content, text="视频合成模式:").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(10, 0))
        mode_frame = tk.Frame(decoding_content)
        mode_frame.grid(row=1, column=1, sticky="w", pady=(10, 0))

        ffmpeg_radio = tk.Radiobutton(mode_frame, text="FFmpeg", variable=self.video_synthesis_mode, value="ffmpeg")
        ffmpeg_radio.pack(side="left")
        opencv_radio = tk.Radiobutton(mode_frame, text="OpenCV", variable=self.video_synthesis_mode, value="opencv")
        opencv_radio.pack(side="left", padx=(10, 0))

        # Video frame rate
        tk.Label(decoding_content, text="视频帧率:").grid(row=2, column=0, sticky="w", padx=(0, 5), pady=(10, 0))
        frame_rate_frame = tk.Frame(decoding_content)
        frame_rate_frame.grid(row=2, column=1, sticky="w", pady=(10, 0))

        frame_rate_scale = tk.Scale(frame_rate_frame, from_=1, to=60, orient="horizontal",
                                   variable=self.video_frame_rate, showvalue=True, length=150)
        frame_rate_scale.pack(side="left")
        tk.Label(frame_rate_frame, text="fps", font=("Arial", 8)).pack(side="left", padx=(5, 0))

        # 应用程序设置分组
        app_settings_content = tk.Frame(accordion.main_frame)

        # Always on top option
        tk.Label(app_settings_content, text="窗口置顶:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        always_on_top_check = tk.Checkbutton(app_settings_content, variable=self.always_on_top,
                                           command=self._on_always_on_top_changed)
        always_on_top_check.grid(row=0, column=1, sticky="w")

        # 添加分组到手风琴控件
        accordion.add_section("通用选项", general_content, is_expanded=False)
        accordion.add_section("编码选项", encoding_content, is_expanded=False)
        accordion.add_section("解码选项", decoding_content, is_expanded=False)
        accordion.add_section("应用程序设置", app_settings_content, is_expanded=False)

        return accordion.main_frame

    def _on_always_on_top_changed(self):
        """置顶状态改变时的回调"""
        if self.on_always_on_top_changed:
            self.on_always_on_top_changed(self.always_on_top.get())

    def get_compression_level(self):
        """获取压缩等级"""
        return self.compression_level.get()

    def get_steganography_width(self):
        """获取隐写图像宽度"""
        return self.steganography_width.get()

    def get_steganography_height(self):
        """获取隐写图像高度"""
        return self.steganography_height.get()

    def get_use_alpha(self):
        """获取是否使用alpha通道"""
        return self.use_alpha.get()

    def get_top_margin_ratio(self):
        """获取上预留区域百分比（返回小数）"""
        return self.top_margin_ratio.get() / 100.0

    def get_bottom_margin_ratio(self):
        """获取下预留区域百分比（返回小数）"""
        return self.bottom_margin_ratio.get() / 100.0

    def get_always_on_top(self):
        """获取是否窗口置顶"""
        return self.always_on_top.get()

    def get_video_synthesis_mode(self):
        """获取视频合成模式"""
        return self.video_synthesis_mode.get()

    def get_video_frame_rate(self):
        """获取视频帧率"""
        return self.video_frame_rate.get()

    def get_image_encryption_method(self):
        """获取图像加密方法"""
        return self.image_encryption_method.get()