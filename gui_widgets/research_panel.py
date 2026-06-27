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
    VAD 维度柱状图

    使用 matplotlib 绘制 Valence-Arousal-Dominance 等维度的水平柱状图，
    比雷达图更清晰易读，支持直接比较各维度数值。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._values = {"valence": 0, "arousal": 0, "dominance": 0}

        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(4, 2.8), dpi=100)
            self.figure.patch.set_facecolor('#F2EDE4')
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.canvas)
        else:
            layout = QVBoxLayout(self)
            label = QLabel("matplotlib 不可用\n无法显示图表")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("color: #8A8580;")
            layout.addWidget(label)

    def update_values(self, valence, arousal, dominance, negative_load=0, uncertainty=0):
        """更新柱状图数据"""
        self._values = {"valence": valence, "arousal": arousal,
                        "dominance": dominance, "negative_load": negative_load,
                        "uncertainty": uncertainty}
        if MATPLOTLIB_AVAILABLE:
            self._draw()

    def _draw(self):
        """绘制水平柱状图 — 构成主义风格"""
        self.ax.clear()

        import numpy as np

        # 维度名称和数值
        categories = ['效价 (Valence)', '唤醒度 (Arousal)', '掌控感 (Dominance)',
                      '负性负荷 (Neg.Load)', '不确定性 (Uncertainty)']

        # 效价范围 -1~1，映射到 -1~1（保持原始尺度以显示正负方向）
        values = [
            self._values['valence'],        # -1 ~ 1
            self._values['arousal'],         # 0 ~ 1
            self._values['dominance'],       # 0 ~ 1
            self._values['negative_load'],   # 0 ~ 1
            self._values['uncertainty']      # 0 ~ 1
        ]

        # 水平柱状图（从下到上显示）
        y_pos = np.arange(len(categories))
        bar_colors = ['#C44B4F', '#C44B4F', '#C44B4F', '#2B2B2B', '#8A8580']

        # 对于效价（可能为负值），使用双向条
        bars = self.ax.barh(y_pos, values, height=0.6, color=bar_colors,
                           edgecolor='#2B2B2B', linewidth=2, zorder=3,
                           alpha=0.85)

        # 效价零点参考线
        self.ax.axvline(x=0, color='#2B2B2B', linewidth=1.5, linestyle='-', alpha=0.5, zorder=2)

        # 标签
        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels(categories, fontsize=9, color='#2B2B2B', fontweight='bold')
        self.ax.set_xlim(-1.05, 1.05)
        self.ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
        self.ax.set_xticklabels(['-1.0', '-0.5', '0', '0.5', '1.0'], fontsize=8, color='#2B2B2B')
        self.ax.set_xlabel("", fontsize=9, color='#2B2B2B')

        # 数值标注
        for i, (v, c) in enumerate(zip(values, categories)):
            if v >= 0:
                x_pos = v + 0.04
                ha = 'left'
            else:
                x_pos = v - 0.04
                ha = 'right'
            self.ax.text(x_pos, i, f'{v:+.2f}', va='center', ha=ha,
                         fontsize=8, fontweight='bold', color='#2B2B2B', zorder=4)

        # 构成主义风格装饰
        self.ax.set_facecolor('#F2EDE4')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('#2B2B2B')
        self.ax.spines['left'].set_linewidth(2)
        self.ax.spines['bottom'].set_color('#2B2B2B')
        self.ax.spines['bottom'].set_linewidth(2)
        self.ax.tick_params(axis='y', length=0)
        self.ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=1, color='#2B2B2B', zorder=1)

        # 标题
        self.ax.set_title("情感维度指标", fontsize=11, fontweight='bold', pad=10, color='#2B2B2B', loc='left')

        self.figure.tight_layout(pad=1.5)
        self.canvas.draw()


class ExportToolbar(QWidget):
    """数据导出工具栏 — 支持 CSV (SPSS/Excel) 和 JSON (完整科研数据) 两种格式"""

    export_csv_clicked = pyqtSignal()
    export_json_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(10)

        title = QLabel("■ 导出研究数据：")
        title.setFont(QFont("Microsoft YaHei", 11, QFont.Black))
        title.setStyleSheet("color: #2B2B2B; padding: 4px 0;")
        layout.addWidget(title)

        csv_btn = QPushButton("导出 CSV (Excel/SPSS)")
        csv_btn.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        csv_btn.setMinimumHeight(36)
        csv_btn.setToolTip("导出为 UTF-8 CSV 格式，可在 Excel/SPSS 中直接打开\n包含所有数值型指标和质量评估数据")
        csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #F2EDE4;
                border: 2px solid #2B2B2B;
                padding: 6px 18px;
                color: #2B2B2B;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C44B4F;
                color: #FFFFFF;
            }
        """)
        csv_btn.clicked.connect(self.export_csv_clicked.emit)
        layout.addWidget(csv_btn)

        json_btn = QPushButton("导出 JSON (完整数据)")
        json_btn.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        json_btn.setMinimumHeight(36)
        json_btn.setToolTip("导出为 JSON 格式，包含原始模型输出、完整概率分布等全部字段\n适合科研复算和深度分析")
        json_btn.setStyleSheet("""
            QPushButton {
                background-color: #2B2B2B;
                border: 2px solid #2B2B2B;
                padding: 6px 18px;
                color: #FFFFFF;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C44B4F;
                color: #FFFFFF;
            }
        """)
        json_btn.clicked.connect(self.export_json_clicked.emit)
        layout.addWidget(json_btn)

        layout.addStretch()
