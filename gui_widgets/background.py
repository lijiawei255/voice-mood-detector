# -*- coding: utf-8 -*-
"""
构成主义几何背景层

提供苏联构成主义风格的装饰性背景绘制：
- 45度交叉斜线纹理（工业蓝图感）
- 大型砖红楔形 + 炭黑三角（El Lissitzky 式）
- 斜穿半透明色条（对角线动态张力）

作者：Jiawei Li
许可证：GPL v3
"""

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QPolygon


class ConstructivistBackground(QWidget):
    """构成主义几何背景层 - 苏联构成主义风格（激进版：大面积楔形、对角分割、工业感）"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._cache = None

    def paintEvent(self, event):
        if self._cache and self._cache.size() == self.size():
            painter = QPainter(self)
            painter.drawPixmap(0, 0, self._cache)
            return

        pixmap = QPixmap(self.size())
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # 纯色背景：暖白色（去掉渐变，更符合构成主义平涂风格）
        p.fillRect(self.rect(), QColor("#F2EDE4"))

        # 大尺度45度斜线纹理（工业蓝图感）
        pen = QPen(QColor(43, 43, 43, 10))
        pen.setWidth(1)
        p.setPen(pen)
        spacing = 48
        for i in range(-h, w + h, spacing):
            p.drawLine(i, 0, i + h, h)
        # 反向45度交叉线网格（更浓的工程图纸感）
        pen2 = QPen(QColor(43, 43, 43, 6))
        pen2.setWidth(1)
        p.setPen(pen2)
        for i in range(0, w + h, spacing * 2):
            p.drawLine(i, 0, i - h, h)

        # 左上角大型砖红楔形（El Lissitzky式红色楔形 — 指向左下 / "击穿"感）
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#C44B4F"))
        wedge_size = min(w, h) // 6
        wedge_tl = QPolygon([QPoint(0, 0), QPoint(wedge_size, 0), QPoint(0, wedge_size)])
        p.drawPolygon(wedge_tl)

        # 右下角炭黑大三角（构成主义标志性元素）
        p.setBrush(QColor("#2B2B2B"))
        tri_size = min(w, h) // 10
        triangle_br = QPolygon([QPoint(w, h), QPoint(w - tri_size, h), QPoint(w, h - tri_size)])
        p.drawPolygon(triangle_br)

        # 左侧纵向红色装饰条（5px宽，从顶部三角延伸至底部1/3处）
        p.setBrush(QColor("#C44B4F"))
        p.drawRect(0, wedge_size, 4, h // 3)

        # 底部红色装饰线（粗横条）
        p.setBrush(QColor("#C44B4F"))
        p.drawRect(0, h - 3, w // 4, 3)

        # 右侧纵向炭黑细线（工业分隔感）
        p.setBrush(QColor("#2B2B2B"))
        p.drawRect(w - 2, h // 5, 2, h * 2 // 3)

        # 大幅面斜穿红色半透明条（构成主义标志性斜向动势） — 提高不透明度
        p.setBrush(QColor(196, 75, 79, 55))  # #C44B4F 约22%透明度
        p.setPen(Qt.NoPen)
        stripe_width = 120
        diag_stripe = QPolygon([
            QPoint(w, h // 5),
            QPoint(w, h // 5 + stripe_width),
            QPoint(w * 3 // 5, h // 5 + w * 2 // 5 + stripe_width),
            QPoint(w * 3 // 5, h // 5 + w * 2 // 5)
        ])
        p.drawPolygon(diag_stripe)
        # 第二道斜向条（反向，炭黑色，更宽）
        p.setBrush(QColor(43, 43, 43, 35))
        stripe2 = QPolygon([
            QPoint(0, h * 5 // 8),
            QPoint(0, h * 5 // 8 + 80),
            QPoint(w * 2 // 5, h * 5 // 8 - w * 2 // 5 + 80),
            QPoint(w * 2 // 5, h * 5 // 8 - w * 2 // 5)
        ])
        p.drawPolygon(stripe2)

        p.end()
        self._cache = pixmap
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._cache)

    def resizeEvent(self, event):
        self._cache = None
        super().resizeEvent(event)
