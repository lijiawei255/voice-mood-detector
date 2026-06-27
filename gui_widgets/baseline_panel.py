# -*- coding: utf-8 -*-
"""
个人基线管理面板 (P2)

提供基线采集、查看、重置的 UI 组件。
极简主义风格。

作者：Jiawei Li
许可证：GPL v3
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QProgressBar, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


# 极简主义色板常量
_TEXT_PRIMARY = "#1D1D1F"
_TEXT_SECONDARY = "#86868B"
_ACCENT = "#1A73E8"
_BORDER = "#D2D2D7"
_BG_TERTIARY = "#EBEBEF"


class BaselinePanel(QGroupBox):
    """个人基线管理面板 — 极简主义风格"""

    baseline_collect_requested = pyqtSignal()
    baseline_reset_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("个人基线管理", parent)
        self.setFont(QFont("Microsoft YaHei", 11, QFont.Medium))
        self.setObjectName("baselineGroup")
        self.setStyleSheet(f"""
            QGroupBox#baselineGroup {{
                border: 1px solid {_BORDER};
                border-radius: 8px;
                margin-top: 16px;
                padding-top: 18px;
                background-color: #FFFFFF;
                font-weight: 500;
            }}
            QGroupBox#baselineGroup::title {{
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
                color: {_TEXT_PRIMARY};
            }}
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 状态指示
        self.status_label = QLabel("基线状态：未建立")
        self.status_label.setFont(QFont("Microsoft YaHei", 11))
        self.status_label.setStyleSheet(f"color: {_TEXT_SECONDARY}; background: transparent;")
        layout.addWidget(self.status_label)

        self.info_label = QLabel(
            "个人基线用于评估相对变化，需要采集 3-5 条"
            "平静状态下的语音样本。首次建立后，后续评估将"
            "同时显示绝对分数和相对基线偏移。"
        )
        self.info_label.setFont(QFont("Microsoft YaHei", 10))
        self.info_label.setStyleSheet(f"color: {_TEXT_SECONDARY}; background: transparent;")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # 进度条
        self.baseline_progress = QProgressBar()
        self.baseline_progress.setRange(0, 3)
        self.baseline_progress.setValue(0)
        self.baseline_progress.setFormat("样本: %v / 3")
        self.baseline_progress.setMinimumHeight(24)
        self.baseline_progress.setTextVisible(True)
        self.baseline_progress.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {_BORDER};
                border-radius: 4px;
                background-color: {_BG_TERTIARY};
                text-align: center;
                color: {_TEXT_PRIMARY};
                font-size: 10pt;
            }}
            QProgressBar::chunk {{
                background-color: {_ACCENT};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self.baseline_progress)

        # 按钮行
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.collect_btn = QPushButton("采集基线样本")
        self.collect_btn.setFont(QFont("Microsoft YaHei", 10))
        self.collect_btn.setMinimumHeight(34)
        self.collect_btn.setCursor(Qt.PointingHandCursor)
        self.collect_btn.setStyleSheet(_primary_btn_style())
        self.collect_btn.clicked.connect(self.baseline_collect_requested.emit)
        btn_layout.addWidget(self.collect_btn)

        self.reset_btn = QPushButton("重置基线")
        self.reset_btn.setFont(QFont("Microsoft YaHei", 10))
        self.reset_btn.setMinimumHeight(34)
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.setStyleSheet(_ghost_btn_style())
        self.reset_btn.clicked.connect(self._confirm_reset)
        self.reset_btn.setEnabled(False)
        btn_layout.addWidget(self.reset_btn)

        layout.addLayout(btn_layout)

        # 偏移显示
        self.deviation_label = QLabel("")
        self.deviation_label.setFont(QFont("Microsoft YaHei", 10))
        self.deviation_label.setStyleSheet(f"color: {_TEXT_PRIMARY}; background: transparent;")
        self.deviation_label.setWordWrap(True)
        self.deviation_label.hide()
        layout.addWidget(self.deviation_label)

    def _confirm_reset(self):
        reply = QMessageBox.question(
            self, "确认重置", "确定要重置个人基线吗？\n\n"
            "所有已采集的基线样本将被清除。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.baseline_reset_requested.emit()

    def update_status(self, baseline_manager):
        """根据基线管理器的状态更新 UI"""
        if baseline_manager is None:
            self.status_label.setText("基线状态：不可用")
            self.status_label.setStyleSheet(f"color: {_TEXT_SECONDARY}; background: transparent;")
            self.baseline_progress.setValue(0)
            self.reset_btn.setEnabled(False)
            return

        n = baseline_manager.get_sample_count()
        is_est = baseline_manager.is_established()

        self.baseline_progress.setValue(min(n, 3))

        if is_est:
            bl = baseline_manager.get_baseline()
            n_total = bl.get("n_samples", n) if bl else n
            self.status_label.setText(f"基线状态：已建立（{n_total} 个样本）")
            self.status_label.setStyleSheet(f"color: {_ACCENT}; font-weight: bold; background: transparent;")
            self.reset_btn.setEnabled(True)
            self.collect_btn.setText("追加基线样本")
        else:
            self.status_label.setText(f"基线状态：采集中（{n}/3）")
            self.status_label.setStyleSheet(f"color: {_TEXT_SECONDARY}; background: transparent;")
            self.reset_btn.setEnabled(n > 0)
            self.collect_btn.setText("采集基线样本")

    def show_deviation(self, deviation_data):
        """显示基线偏移信息"""
        if not deviation_data or not deviation_data.get("available"):
            self.deviation_label.hide()
            return

        self.deviation_label.setText(
            f"{deviation_data.get('summary', '')}"
        )
        self.deviation_label.show()


def _primary_btn_style():
    return f"""
        QPushButton {{
            background-color: {_ACCENT};
            border: none;
            border-radius: 6px;
            padding: 6px 16px;
            color: #FFFFFF;
        }}
        QPushButton:hover {{
            background-color: #1557B0;
        }}
        QPushButton:disabled {{
            background-color: {_BORDER};
            color: {_TEXT_SECONDARY};
        }}
    """


def _ghost_btn_style():
    return f"""
        QPushButton {{
            background-color: transparent;
            border: 1px solid {_BORDER};
            border-radius: 6px;
            padding: 6px 16px;
            color: {_TEXT_PRIMARY};
        }}
        QPushButton:hover {{
            border-color: {_ACCENT};
            color: {_ACCENT};
        }}
        QPushButton:disabled {{
            color: {_TEXT_SECONDARY};
            border-color: {_BORDER};
        }}
    """
