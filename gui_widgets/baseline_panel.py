# -*- coding: utf-8 -*-
"""
个人基线管理面板 (P2)

提供基线采集、查看、重置的 UI 组件。

作者：Jiawei Li
许可证：GPL v3
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QProgressBar, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


class BaselinePanel(QGroupBox):
    """个人基线管理面板"""

    baseline_collect_requested = pyqtSignal()
    baseline_reset_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("个人基线管理", parent)
        self.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.setObjectName("baselineGroup")
        self.setStyleSheet("""
            QGroupBox#baselineGroup {
                border: 3px solid #2B2B2B;
                margin-top: 12px;
                padding-top: 20px;
                background-color: #F7F5F2;
            }
            QGroupBox#baselineGroup::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 0 12px;
                color: #2B2B2B;
            }
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 状态指示
        self.status_label = QLabel("基线状态：未建立")
        self.status_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.status_label.setStyleSheet("color: #E67E22;")
        layout.addWidget(self.status_label)

        self.info_label = QLabel(
            "个人基线用于评估相对变化，需要采集 3-5 条"
            "平静状态下的语音样本。首次建立后，后续评估将"
            "同时显示绝对分数和相对基线偏移。"
        )
        self.info_label.setFont(QFont("Microsoft YaHei", 10))
        self.info_label.setStyleSheet("color: #8A8580;")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # 进度条
        self.baseline_progress = QProgressBar()
        self.baseline_progress.setRange(0, 3)
        self.baseline_progress.setValue(0)
        self.baseline_progress.setFormat("样本: %v / 3")
        self.baseline_progress.setMinimumHeight(24)
        self.baseline_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #2B2B2B;
                background-color: #E8E4DF;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #C44B4F;
            }
        """)
        layout.addWidget(self.baseline_progress)

        # 按钮行
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.collect_btn = QPushButton("采集基线样本")
        self.collect_btn.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.collect_btn.setMinimumHeight(36)
        self.collect_btn.setStyleSheet(_btn_style())
        self.collect_btn.clicked.connect(self.baseline_collect_requested.emit)
        btn_layout.addWidget(self.collect_btn)

        self.reset_btn = QPushButton("重置基线")
        self.reset_btn.setFont(QFont("Microsoft YaHei", 11))
        self.reset_btn.setMinimumHeight(36)
        self.reset_btn.setStyleSheet(_btn_style())
        self.reset_btn.clicked.connect(self._confirm_reset)
        self.reset_btn.setEnabled(False)
        btn_layout.addWidget(self.reset_btn)

        layout.addLayout(btn_layout)

        # 偏移显示
        self.deviation_label = QLabel("")
        self.deviation_label.setFont(QFont("Microsoft YaHei", 10))
        self.deviation_label.setStyleSheet("color: #2B2B2B;")
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
            self.status_label.setStyleSheet("color: #95A5A6;")
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
            self.status_label.setStyleSheet("color: #27AE60;")
            self.reset_btn.setEnabled(True)
            self.collect_btn.setText("追加基线样本")
        else:
            self.status_label.setText(f"基线状态：采集中（{n}/3）")
            self.status_label.setStyleSheet("color: #E67E22;")
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


def _btn_style():
    return """
        QPushButton {
            background-color: #F7F5F2;
            border: 2px solid #2B2B2B;
            padding: 4px 16px;
            color: #2B2B2B;
        }
        QPushButton:hover {
            background-color: #E8E4DF;
        }
        QPushButton:disabled {
            color: #C4C0BC;
            border-color: #C4C0BC;
        }
    """
