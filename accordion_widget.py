"""
手风琴控件模块
提供可折叠的分组控件
"""
import tkinter as tk


class AccordionSection:
    """手风琴控件的单个分组"""

    def __init__(self, parent, title, content_frame, is_expanded=True):
        self.parent = parent
        self.title = title
        self.content_frame = content_frame
        self.is_expanded = is_expanded
        self.header_frame = None
        self.toggle_button = None

        self._create_header()
        self._update_visibility()

    def _create_header(self):
        """创建分组标题栏"""
        self.header_frame = tk.Frame(self.parent, relief="raised", bd=1)
        self.header_frame.pack(fill="x", padx=5, pady=(5, 0))

        # 切换按钮
        self.toggle_button = tk.Button(
            self.header_frame,
            text="▼ " + self.title if self.is_expanded else "▶ " + self.title,
            font=("Arial", 10, "bold"),
            command=self.toggle,
            relief="flat",
            bg="#f0f0f0",
            activebackground="#e0e0e0",
            anchor="w",  # 文本左对齐
            padx=10  # 增加水平内边距
        )
        self.toggle_button.pack(fill="x", expand=True)  # 填满整个标题栏宽度

    def toggle(self):
        """切换分组的展开/折叠状态"""
        self.is_expanded = not self.is_expanded
        self._update_visibility()
        self._update_button_text()

    def _update_visibility(self):
        """更新内容区域的可见性"""
        if self.is_expanded:
            # 获取header_frame在父容器中的索引位置
            header_index = self.parent.pack_slaves().index(self.header_frame)
            # 将内容框架插入到header_frame之后的位置
            self.content_frame.pack(fill="x", padx=10, pady=(0, 5), before=self.parent.pack_slaves()[header_index + 1] if header_index + 1 < len(self.parent.pack_slaves()) else None)
        else:
            self.content_frame.pack_forget()

    def _update_button_text(self):
        """更新按钮文本"""
        if self.is_expanded:
            self.toggle_button.config(text="▼ " + self.title)
        else:
            self.toggle_button.config(text="▶ " + self.title)


class AccordionWidget:
    """手风琴控件主类"""

    def __init__(self, parent):
        self.parent = parent
        self.sections = []

        # 创建主容器
        self.main_frame = tk.Frame(parent)
        self.main_frame.pack(fill="x", padx=10, pady=10)

    def add_section(self, title, content_frame, is_expanded=True):
        """添加一个分组"""
        section = AccordionSection(self.main_frame, title, content_frame, is_expanded)
        self.sections.append(section)
        return section

    def expand_all(self):
        """展开所有分组"""
        for section in self.sections:
            if not section.is_expanded:
                section.toggle()

    def collapse_all(self):
        """折叠所有分组"""
        for section in self.sections:
            if section.is_expanded:
                section.toggle()