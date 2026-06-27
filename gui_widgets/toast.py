# -*- coding: utf-8 -*-
"""
Toast 通知组件模块

提供非阻塞式弹窗通知：
- ToastNotification: 单个通知弹窗（极简主义风格）
- ToastManager: 通知管理器（防止重叠）

极简主义风格：圆角卡片 + 左侧色条（错误红/警告黄/信息蓝）

作者：Jiawei Li
许可证：GPL v3
"""

from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPainter, QBrush, QPainterPath, QColor


# 极简主义色板
_TEXT_PRIMARY = "#1D1D1F"
_TEXT_SECONDARY = "#86868B"
_ACCENT = "#1A73E8"
_WARNING = "#F9AB00"
_ERROR = "#EA4335"
_BORDER = "#D2D2D7"


class ToastNotification(QFrame):
    """
    极简主义风格弹窗通知组件

    用于在界面右上角展示错误/警告信息，支持自动消失和手动关闭。
    圆角卡片 + 左侧色条编码级别。
    """

    def __init__(self, parent, message, level="warning", duration=5000):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SubWindow)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedWidth(380)
        self._opacity = 1.0
        self._level = level

        # 级别色彩编码 — 极简主义
        if level == "error":
            title_text = "错误"
            stripe_color = _ERROR
        else:
            title_text = "警告"
            stripe_color = _WARNING

        # 布局：左侧色条 + 内容
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 左侧色条（4px 宽，圆角）
        self._stripe_color = QColor(stripe_color)
        stripe = QFrame()
        stripe.setFixedWidth(4)
        stripe.setStyleSheet(f"background-color: {stripe_color}; border-top-left-radius: 8px; border-bottom-left-radius: 8px;")
        main_layout.addWidget(stripe)

        # 内容区
        content_wrap = QFrame()
        content_wrap.setStyleSheet(f"background-color: #FFFFFF; border-top-right-radius: 8px; border-bottom-right-radius: 8px;")
        content_layout = QHBoxLayout(content_wrap)
        content_layout.setContentsMargins(16, 12, 12, 12)
        content_layout.setSpacing(10)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)

        title_label = QLabel(title_text)
        title_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        title_label.setStyleSheet(f"color: {_TEXT_PRIMARY}; background: transparent;")
        text_layout.addWidget(title_label)

        msg_label = QLabel(message)
        msg_label.setFont(QFont("Microsoft YaHei", 9))
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet(f"color: {_TEXT_SECONDARY}; background: transparent;")
        msg_label.setMaximumWidth(300)
        text_layout.addWidget(msg_label)

        content_layout.addLayout(text_layout, 1)

        # 关闭按钮
        close_btn = QPushButton("×")
        close_btn.setFixedSize(24, 24)
        close_btn.setFont(QFont("Arial", 14))
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {_TEXT_SECONDARY};
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: #E8EAED;
                color: {_TEXT_PRIMARY};
            }}
        """)
        close_btn.clicked.connect(self.fade_out)
        content_layout.addWidget(close_btn, 0, Qt.AlignTop)

        main_layout.addWidget(content_wrap, 1)

        self.setStyleSheet(f"""
            ToastNotification {{
                background-color: #FFFFFF;
                border: 1px solid {_BORDER};
                border-radius: 8px;
            }}
        """)

        self.adjustSize()

        # 自动消失定时器
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.fade_out)
        self._timer.start(duration)

    def fade_out(self):
        """ 淡出动画后关闭 """
        self._timer.stop()
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
