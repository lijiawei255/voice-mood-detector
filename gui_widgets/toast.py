# -*- coding: utf-8 -*-
"""
Toast 通知组件模块

提供非阻塞式弹窗通知：
- ToastNotification: 单个通知弹窗（构成主义风格）
- ToastManager: 通知管理器（防止重叠）

作者：Jiawei Li
许可证：GPL v3
"""

from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPainter, QBrush, QPainterPath


class ToastNotification(QFrame):
    """
    苹果风格弹窗通知组件

    用于在界面右上角展示错误/警告信息，支持自动消失和手动关闭。
    """

    def __init__(self, parent, message, level="warning", duration=5000):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SubWindow)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedWidth(380)
        self._opacity = 1.0

        # 图标和颜色 - 构成主义风格
        if level == "error":
            icon = "■"  # 几何方块
            bg_color = "#F2EDE4"
            border_color = "#C44B4F"
            title_color = "#C44B4F"
            title_text = "错误"
        else:
            icon = "▲"  # 几何三角
            bg_color = "#F2EDE4"
            border_color = "#8A8580"
            title_color = "#2B2B2B"
            title_text = "警告"

        # 布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 12, 12)
        main_layout.setSpacing(10)

        # 图标
        icon_label = QLabel(icon)
        icon_label.setFont(QFont("Segoe UI Emoji", 16))
        icon_label.setFixedSize(32, 32)
        icon_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(icon_label)

        # 内容区
        content_layout = QVBoxLayout()
        content_layout.setSpacing(2)

        title_label = QLabel(title_text)
        title_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        title_label.setStyleSheet(f"color: {title_color}; background: transparent;")
        content_layout.addWidget(title_label)

        msg_label = QLabel(message)
        msg_label.setFont(QFont("Microsoft YaHei", 9))
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet("color: #2B2B2B; background: transparent;")
        msg_label.setMaximumWidth(280)
        content_layout.addWidget(msg_label)

        main_layout.addLayout(content_layout, 1)

        # 关闭按钮
        close_btn = QPushButton("×")
        close_btn.setFixedSize(24, 24)
        close_btn.setFont(QFont("Arial", 14))
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #2B2B2B;
                border-radius: 0px;
            }
            QPushButton:hover {
                background: rgba(43,43,43,0.1);
                color: #C44B4F;
            }
        """)
        close_btn.clicked.connect(self.fade_out)
        main_layout.addWidget(close_btn, 0, Qt.AlignTop)

        self.setStyleSheet(f"""
            ToastNotification {{
                background-color: {bg_color};
                border: 3px solid {border_color};
                border-radius: 0px;
            }}
        """)

        # 构成主义风格：无阴影，纯平面，用粗边框强化层次

        self.adjustSize()

        # 自动消失定时器
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.fade_out)
        self._timer.start(duration)

    def fade_out(self):
        """ 淡出动画后关闭 """
        self._timer.stop()
        # 用一个简单的定时器动画模拟淡出
        self._fade_timer = QTimer(self)
        self._fade_step = 0
        self._fade_timer.timeout.connect(self._do_fade)
        self._fade_timer.start(30)

    def _do_fade(self):
        self._fade_step += 1
        self._opacity = max(0, 1.0 - self._fade_step * 0.1)
        self.setWindowOpacity(self._opacity) if hasattr(self, 'setWindowOpacity') else None
        self.update()
        if self._fade_step >= 10:
            self._fade_timer.stop()
            self.close()
            self.deleteLater()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setOpacity(self._opacity)
        path = QPainterPath()
        path.addRect(0, 0, self.width(), self.height())
        painter.fillPath(path, QBrush(self.palette().window().color()))
        super().paintEvent(event)


class ToastManager:
    """
    Toast通知管理器

    管理多个Toast的显示位置，避免重叠。
    """
    def __init__(self, parent):
        self.parent = parent
        self.toasts = []

    def show_toast(self, message, level="warning", duration=5000):
        """ 显示一个Toast通知 """
        # 清理已关闭的toast
        self.toasts = [t for t in self.toasts if t.isVisible()]

        toast = ToastNotification(self.parent, message, level, duration)
        self.toasts.append(toast)

        # 计算位置（右上角堆叠）
        self._reposition()
        toast.show()

    def _reposition(self):
        """ 重新计算所有toast位置 """
        margin_top = 20
        margin_right = 20
        spacing = 10
        y_offset = margin_top

        parent_rect = self.parent.rect()
        for toast in self.toasts:
            if toast.isVisible():
                x = parent_rect.width() - toast.width() - margin_right
                toast.move(x, y_offset)
                y_offset += toast.height() + spacing
