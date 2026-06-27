# -*- coding: utf-8 -*-
"""
背景层 — 极简主义风格

历史上的构成主义几何背景（45度斜线、楔形、三角）已移除。
保留 ConstructivistBackground 类名以避免破坏外部引用，
但绘制逻辑已简化为纯白背景（极简主义的核心：去除装饰）。

作者：Jiawei Li
许可证：GPL v3
"""

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor


class ConstructivistBackground(QWidget):
    """
    背景层 - 极简主义纯白背景

    保留类名（被 gui.py 引用），但不再绘制任何几何装饰。
    仅填充纯白底色，符合瑞士/包豪斯极简风格。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def paintEvent(self, event):
        painter = QPainter(self)
        # 纯白背景 — 极简主义
        painter.fillRect(self.rect(), QColor("#FFFFFF"))
        painter.end()
