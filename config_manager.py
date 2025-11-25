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

    def create_config_frame(self, parent):
        """创建配置选项框架"""
        # 创建手风琴控件
        accordion = AccordionWidget(parent)

        # 编码选项分组
        encoding_content = tk.Frame(accordion.main_frame)

        # Compression level
        tk.Label(encoding_content, text="压缩等级:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        compression_frame = tk.Frame(encoding_content)
        compression_frame.grid(row=0, column=1, sticky="w")

        compression_scale = tk.Scale(compression_frame, from_=-1, to=9, orient="horizontal",
                                   variable=self.compression_level, showvalue=True, length=200)
        compression_scale.pack(side="left")
        tk.Label(compression_frame, text="(-1=默认, 0=无压缩, 9=最大压缩)", font=("Arial", 8)).pack(side="left", padx=(5, 0))

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

        # Margin ratios
        tk.Label(encoding_content, text="上预留区域:").grid(row=3, column=0, sticky="w", padx=(0, 5), pady=(10, 0))
        top_margin_frame = tk.Frame(encoding_content)
        top_margin_frame.grid(row=3, column=1, sticky="w", pady=(10, 0))

        top_margin_scale = tk.Scale(top_margin_frame, from_=0, to=100, orient="horizontal",
                                   variable=self.top_margin_ratio, showvalue=True, length=150, resolution=1)
        top_margin_scale.pack(side="left")
        tk.Label(top_margin_frame, text="%", font=("Arial", 8)).pack(side="left", padx=(5, 0))

        tk.Label(encoding_content, text="下预留区域:").grid(row=4, column=0, sticky="w", padx=(0, 5), pady=(10, 0))
        bottom_margin_frame = tk.Frame(encoding_content)
        bottom_margin_frame.grid(row=4, column=1, sticky="w", pady=(10, 0))

        bottom_margin_scale = tk.Scale(bottom_margin_frame, from_=0, to=100, orient="horizontal",
                                      variable=self.bottom_margin_ratio, showvalue=True, length=150, resolution=1)
        bottom_margin_scale.pack(side="left")
        tk.Label(bottom_margin_frame, text="%", font=("Arial", 8)).pack(side="left", padx=(5, 0))

        # 应用程序设置分组
        app_settings_content = tk.Frame(accordion.main_frame)

        # Always on top option
        tk.Label(app_settings_content, text="窗口置顶:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        always_on_top_check = tk.Checkbutton(app_settings_content, variable=self.always_on_top,
                                           command=self._on_always_on_top_changed)
        always_on_top_check.grid(row=0, column=1, sticky="w")

        # 添加分组到手风琴控件
        accordion.add_section("编码选项", encoding_content, is_expanded=True)
        accordion.add_section("应用程序设置", app_settings_content, is_expanded=True)

        return accordion.main_frame

    def _on_always_on_top_changed(self):
        """置顶状态改变时的回调"""
        if self.on_always_on_top_changed:
            self.on_always_on_top_changed(self.always_on_top.get())

    def get_compression_level(self):
        """获取压缩等级"""
        return self.compression_level.get()

    def get_noise_width(self):
        """获取隐写图像宽度"""
        return self.steganography_width.get()

    def get_noise_height(self):
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