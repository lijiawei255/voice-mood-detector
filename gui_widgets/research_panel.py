# -*- coding: utf-8 -*-
"""
科研评估模式面板组件 (P1)

提供科研模式专用的UI组件：
- ResearchModePanel: 科研模式控制面板（环境检测、质量门控、多段录音）
- ResearchRadarChart: VAD 维度雷达图 (matplotlib)
- ExportToolbar: 数据导出工具栏

作者：Jiawei Li
许可证：GPL v3
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFileDialog, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib
    matplotlib.use('Qt5Agg')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ResearchRadarChart(QWidget):
    """
    VAD 维度雷达图

    使用 matplotlib 绘制 Valence-Arousal-Dominance 三维雷达图，
    叠加负性负荷和情绪不确定性作为参考指标。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._values = {"valence": 0, "arousal": 0, "dominance": 0}

        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(3, 3), dpi=80)
            self.figure.patch.set_facecolor('#F2EDE4')
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111, polar=True)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.canvas)
        else:
            layout = QVBoxLayout(self)
            label = QLabel("matplotlib 不可用\n无法显示雷达图")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("color: #8A8580;")
            layout.addWidget(label)

    def update_values(self, valence, arousal, dominance, negative_load=0, uncertainty=0):
        """更新雷达图数据"""
        self._values = {"valence": valence, "arousal": arousal,
                        "dominance": dominance, "negative_load": negative_load,
                        "uncertainty": uncertainty}
        if MATPLOTLIB_AVAILABLE:
            self._draw()

    def _draw(self):
        """绘制雷达图 — 构成主义风格"""
        self.ax.clear()

        import numpy as np

        categories = ['效价\nValence', '唤醒度\nArousal', '掌控感\nDominance',
                      '负性负荷\nNeg.Load', '不确定性\nUncertainty']

        # 效价范围 -1~1，需要映射到 0~1
        valence_mapped = (self._values['valence'] + 1) / 2
        values = [
            valence_mapped,
            self._values['arousal'],
            self._values['dominance'],
            self._values['negative_load'],
            self._values['uncertainty']
        ]

        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        values_plot = values + values[:1]

        # 绘制
        self.ax.set_theta_offset(np.pi / 2)
        self.ax.set_theta_direction(-1)

        self.ax.set_xticks(angles[:-1])
        self.ax.set_xticklabels(categories, fontsize=9, color='#2B2B2B', fontweight='bold',
                                fontfamily='sans-serif')

        self.ax.set_ylim(0, 1)
        self.ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        self.ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=8, color='#2B2B2B',
                                fontweight='bold')
        self.ax.set_rlabel_position(30)

        # 网格线 - 构成主义粗线风格
        self.ax.yaxis.grid(True, color='#2B2B2B', linewidth=1.5, linestyle='-', alpha=0.6)
        self.ax.xaxis.grid(True, color='#2B2B2B', linewidth=1.5, linestyle='-', alpha=0.6)

        # 填充区域 + 粗线边框
        self.ax.fill(angles, values_plot, alpha=0.3, color='#C44B4F')
        self.ax.plot(angles, values_plot, linewidth=3, color='#C44B4F', solid_capstyle='round')

        self.ax.set_facecolor('#F2EDE4')
        self.ax.spines['polar'].set_color('#2B2B2B')
        self.ax.spines['polar'].set_linewidth(2.5)

        self.canvas.draw()


class ExportToolbar(QWidget):
    """数据导出工具栏"""

    export_csv_clicked = pyqtSignal()
    export_json_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QLabel("导出研究数据：")
        title.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        title.setStyleSheet("color: #2B2B2B;")
        layout.addWidget(title)

        csv_btn = QPushButton("导出 CSV")
        csv_btn.setFont(QFont("Microsoft YaHei", 10))
        csv_btn.setMinimumHeight(32)
        csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #F2EDE4;
                border: 2px solid #2B2B2B;
                padding: 4px 16px;
                color: #2B2B2B;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E8E3DA;
            }
        """)
        csv_btn.clicked.connect(self.export_csv_clicked.emit)
        layout.addWidget(csv_btn)

        json_btn = QPushButton("导出 JSON")
        json_btn.setFont(QFont("Microsoft YaHei", 10))
        json_btn.setMinimumHeight(32)
        json_btn.setStyleSheet("""
            QPushButton {
                background-color: #F2EDE4;
                border: 2px solid #2B2B2B;
                padding: 4px 16px;
                color: #2B2B2B;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E8E3DA;
            }
        """)
        json_btn.clicked.connect(self.export_json_clicked.emit)
        layout.addWidget(json_btn)

        layout.addStretch()
