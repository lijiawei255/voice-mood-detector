# -*- coding: utf-8 -*-
"""
分数卡片控件 — 极简主义风格

自定义的分数展示卡片控件，用于在结果区域展示：
- 情绪稳定度分数
- 情绪状态等级
- 主要情绪

设计特点（极简主义 / 瑞士风格）：
- 浅灰背景 + 细线边框，无装饰图案
- 等宽粗体数字展示为视觉焦点
- 支持自定义颜色和副标题
- 适度圆角（4px）增加柔和感

作者：Jiawei Li
许可证：GPL v3
"""

from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from .fonts import UI_FONT, MONO_FONT


class ScoreCard(QFrame):
    """
    分数卡片控件类 - 极简主义风格

    参数：
        title (str): 卡片标题
        parent: 父窗口部件
        accent_color (str): 装饰色（蓝色，用于值的高亮，默认 #1A73E8）
        compact (bool): 是否为紧凑模式（用于右侧小卡片）
    """
    def __init__(self, title, parent=None, accent_color="#1A73E8", compact=False):
        super().__init__(parent)
        self.setObjectName("scoreCard")
        self._accent_color = accent_color
        self._compact = compact

        if compact:
            self.setMinimumHeight(104)
            self.setMaximumHeight(140)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            title_font = QFont(UI_FONT, 11, QFont.Medium)
            value_font = QFont(MONO_FONT, 30, QFont.Bold)
            sub_font = QFont(UI_FONT, 10)
            margins = (16, 14, 16, 12)
            spacing = 6
            title_h = 22
            value_h = 40
            sub_h = 18
        else:
            self.setMinimumHeight(200)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            title_font = QFont(UI_FONT, 12, QFont.Medium)
            value_font = QFont(MONO_FONT, 42, QFont.Bold)
            sub_font = QFont(UI_FONT, 12)
            margins = (24, 24, 24, 24)
            spacing = 10
            title_h = 26
            value_h = 66
            sub_h = 26

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*margins)
        layout.setSpacing(spacing)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("cardTitle")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(title_font)
        self.title_label.setMinimumHeight(title_h)
        layout.addWidget(self.title_label)

        self.value_label = QLabel("--")
        self.value_label.setObjectName("cardValue")
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setFont(value_font)
        self.value_label.setMinimumHeight(value_h)
        self.value_label.setWordWrap(False)
        layout.addWidget(self.value_label, 1)

        self.sub_label = QLabel("")
        self.sub_label.setObjectName("cardSub")
        self.sub_label.setAlignment(Qt.AlignCenter)
        self.sub_label.setFont(sub_font)
        self.sub_label.setMinimumHeight(sub_h)
        layout.addWidget(self.sub_label)

    def set_value(self, value, color=None, sub_text=""):
        self.value_label.setText(str(value))
        if color:
            self.value_label.setStyleSheet(f"color: {color};")
        if sub_text:
            self.sub_label.setText(sub_text)
            self.sub_label.setVisible(True)
        else:
            self.sub_label.setVisible(False)

    def reset(self):
        self.value_label.setText("--")
        self.value_label.setStyleSheet("")
        self.sub_label.setText("")
        self.sub_label.setVisible(False)
