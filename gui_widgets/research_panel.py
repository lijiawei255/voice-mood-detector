# -*- coding: utf-8 -*-
"""
科研评估模式面板组件 (P1)

提供科研模式专用的UI组件：
- ResearchRadarChart: VAD 维度柱状图 (matplotlib)
- ExportToolbar: 数据导出工具栏

设计风格：极简主义（瑞士/包豪斯）— matplotlib 配色蓝色系、ghost 按钮

作者：Jiawei Li
许可证：GPL v3
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib
    matplotlib.use('Qt5Agg')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# 极简主义色板常量
_BG = "#FFFFFF"
_TEXT_PRIMARY = "#1D1D1F"
_TEXT_SECONDARY = "#86868B"
_ACCENT = "#1A73E8"
_BORDER = "#D2D2D7"


class ResearchRadarChart(QWidget):
    """
    VAD 维度柱状图

    使用 matplotlib 绘制 Valence-Arousal-Dominance 等维度的水平柱状图，
    极简主义风格：白色背景、蓝色数据条、细网格线。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._values = {"valence": 0, "arousal": 0, "dominance": 0}

        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(4, 2.8), dpi=100)
            self.figure.patch.set_facecolor(_BG)
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.canvas)
        else:
            layout = QVBoxLayout(self)
            label = QLabel("matplotlib 不可用\n无法显示图表")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet(f"color: {_TEXT_SECONDARY};")
            layout.addWidget(label)

    def update_values(self, valence, arousal, dominance, negative_load=0, uncertainty=0):
        """更新柱状图数据"""
        self._values = {"valence": valence, "arousal": arousal,
                        "dominance": dominance, "negative_load": negative_load,
                        "uncertainty": uncertainty}
        if MATPLOTLIB_AVAILABLE:
            self._draw()

    def _draw(self):
        """绘制水平柱状图 — 极简主义风格"""
        self.ax.clear()

        import numpy as np

        # 维度名称和数值
        categories = ['效价 (Valence)', '唤醒度 (Arousal)', '掌控感 (Dominance)',
                      '负性负荷 (Neg.Load)', '不确定性 (Uncertainty)']

        values = [
            self._values['valence'],        # -1 ~ 1
            self._values['arousal'],         # 0 ~ 1
            self._values['dominance'],       # 0 ~ 1
            self._values['negative_load'],   # 0 ~ 1
            self._values['uncertainty']      # 0 ~ 1
        ]

        # 水平柱状图（从下到上显示）— 极简蓝色系
        y_pos = np.arange(len(categories))
        bar_colors = [_ACCENT, _ACCENT, _ACCENT, _TEXT_SECONDARY, _TEXT_SECONDARY]

        bars = self.ax.barh(y_pos, values, height=0.5, color=bar_colors,
                           edgecolor='none', zorder=3, alpha=0.9)

        # 效价零点参考线（细灰线）
        self.ax.axvline(x=0, color=_BORDER, linewidth=1, linestyle='-', zorder=2)

        # 标签
        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels(categories, fontsize=9, color=_TEXT_PRIMARY)
        self.ax.set_xlim(-1.05, 1.05)
        self.ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
        self.ax.set_xticklabels(['-1.0', '-0.5', '0', '0.5', '1.0'], fontsize=8, color=_TEXT_SECONDARY)
        self.ax.set_xlabel("", fontsize=9, color=_TEXT_SECONDARY)

        # 数值标注
        for i, (v, c) in enumerate(zip(values, categories)):
            if v >= 0:
                x_pos = v + 0.04
                ha = 'left'
            else:
                x_pos = v - 0.04
                ha = 'right'
            self.ax.text(x_pos, i, f'{v:+.2f}', va='center', ha=ha,
                         fontsize=8, color=_TEXT_PRIMARY, zorder=4)

        # 极简主义样式：隐藏顶/右边框，左侧细线
        self.ax.set_facecolor(_BG)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color(_BORDER)
        self.ax.spines['left'].set_linewidth(1)
        self.ax.spines['bottom'].set_color(_BORDER)
        self.ax.spines['bottom'].set_linewidth(1)
        self.ax.tick_params(axis='y', length=0)
        self.ax.grid(True, axis='x', alpha=0.5, linestyle='-', linewidth=0.6, color=_BORDER, zorder=1)

        # 标题
        self.ax.set_title("情感维度指标", fontsize=11, color=_TEXT_PRIMARY, loc='left', pad=10)

        self.figure.tight_layout(pad=1.5)
        self.canvas.draw()


class ExportToolbar(QWidget):
    """数据导出工具栏 — 极简 ghost 风格，支持 CSV (SPSS/Excel) 和 JSON 两种格式"""

    export_csv_clicked = pyqtSignal()
    export_json_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(10)

        title = QLabel("导出研究数据")
        title.setFont(QFont("Microsoft YaHei", 11, QFont.Medium))
        title.setStyleSheet(f"color: {_TEXT_PRIMARY};")
        layout.addWidget(title)

        csv_btn = QPushButton("导出 CSV (Excel/SPSS)")
        csv_btn.setFont(QFont("Microsoft YaHei", 10))
        csv_btn.setMinimumHeight(36)
        csv_btn.setCursor(Qt.PointingHandCursor)
        csv_btn.setToolTip("导出为 UTF-8 CSV 格式，可在 Excel/SPSS 中直接打开\n包含所有数值型指标和质量评估数据")
        csv_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {_BG};
                border: 1px solid {_BORDER};
                border-radius: 6px;
                padding: 6px 18px;
                color: {_TEXT_PRIMARY};
            }}
            QPushButton:hover {{
                border-color: {_ACCENT};
                color: {_ACCENT};
            }}
        """)
        csv_btn.clicked.connect(self.export_csv_clicked.emit)
        layout.addWidget(csv_btn)

        json_btn = QPushButton("导出 JSON (完整数据)")
        json_btn.setFont(QFont("Microsoft YaHei", 10))
        json_btn.setMinimumHeight(36)
        json_btn.setCursor(Qt.PointingHandCursor)
        json_btn.setToolTip("导出为 JSON 格式，包含原始模型输出、完整概率分布等全部字段\n适合科研复算和深度分析")
        json_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {_ACCENT};
                border: 1px solid {_ACCENT};
                border-radius: 6px;
                padding: 6px 18px;
                color: #FFFFFF;
            }}
            QPushButton:hover {{
                background-color: #1557B0;
                border-color: #1557B0;
            }}
        """)
        json_btn.clicked.connect(self.export_json_clicked.emit)
        layout.addWidget(json_btn)

        layout.addStretch()
