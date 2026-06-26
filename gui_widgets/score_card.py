# -*- coding: utf-8 -*-
"""
分数卡片控件 — 苏联构成主义风格

自定义的分数展示卡片控件，用于在结果区域展示：
- 情绪稳定度分数
- 情绪状态等级
- 主要情绪

设计特点：
- 构成主义风格：粗炭黑边框、直角、红色楔形装饰
- 等宽粗体数字展示为视觉焦点
- 支持自定义颜色和副标题

作者：Jiawei Li
许可证：GPL v3
"""

from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QPolygon


class ScoreCard(QFrame):
    """
    分数卡片控件类 - 苏联构成主义风格

    参数：
        title (str): 卡片标题
        parent: 父窗口部件
        accent_color (str): 装饰色（砖红）
    """
    def __init__(self, title, parent=None, accent_color="#C44B4F"):
        super().__init__(parent)
        self.setObjectName("scoreCard")
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._accent_color = accent_color

        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 22, 25, 22)
        layout.setSpacing(10)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("cardTitle")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.title_label.setMinimumHeight(25)
        layout.addWidget(self.title_label)

        self.value_label = QLabel("--")
        self.value_label.setObjectName("cardValue")
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setFont(QFont("Consolas", 38, QFont.Bold))
        self.value_label.setMinimumHeight(60)
        self.value_label.setWordWrap(False)
        layout.addWidget(self.value_label, 1)

        self.sub_label = QLabel("")
        self.sub_label.setObjectName("cardSub")
        self.sub_label.setAlignment(Qt.AlignCenter)
        self.sub_label.setFont(QFont("Microsoft YaHei", 11))
        self.sub_label.setMinimumHeight(25)
        layout.addWidget(self.sub_label)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        # 炭黑粗边框
        pen = QPen(QColor("#2B2B2B"))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawRect(2, 2, self.width()-4, self.height()-4)
        # 左上角大型红色楔形（指向右下）
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(self._accent_color))
        wedge = QPolygon([QPoint(0, 0), QPoint(25, 0), QPoint(0, 25)])
        painter.drawPolygon(wedge)
        # 右下角炭黑小三角
        painter.setBrush(QColor("#2B2B2B"))
        tri = QPolygon([QPoint(self.width(), self.height()),
                       QPoint(self.width() - 12, self.height()),
                       QPoint(self.width(), self.height() - 12)])
        painter.drawPolygon(tri)
        painter.end()
        super().paintEvent(event)

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
