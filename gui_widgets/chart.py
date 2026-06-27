# -*- coding: utf-8 -*-
"""
趋势图表模块 — 极简主义风格

封装 matplotlib 的 FigureCanvas，用于在 PyQt5 界面中显示情绪稳定度趋势图。
极简主义风格：白色背景、蓝色线条、细网格、柔和色带。

作者：Jiawei Li
许可证：GPL v3
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# 极简主义色板
_BG = "#FFFFFF"
_TEXT_PRIMARY = "#1D1D1F"
_TEXT_SECONDARY = "#86868B"
_ACCENT = "#1A73E8"
_SUCCESS = "#34A853"
_WARNING = "#F9AB00"
_ERROR = "#EA4335"
_BORDER = "#D2D2D7"


class MplCanvas(FigureCanvas):
    """
    matplotlib 图表画布类

    封装 matplotlib 的 FigureCanvas，用于在 PyQt5 界面中显示图表。
    主要功能：
    - 绘制情绪稳定度变化趋势图
    - 自动配置中文字体（微软雅黑/黑体）
    - 支持分区域着色（不同稳定度等级不同颜色）
    - 极简主义风格

    参数：
        parent: 父窗口部件
        width: 图表宽度（英寸）
        height: 图表高度（英寸）
        dpi: 图表分辨率
    """
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        if MATPLOTLIB_AVAILABLE:
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            self.axes = self.fig.add_subplot(111)
            super().__init__(self.fig)
            self.setParent(parent)
            self.setup_chinese_font()
            self.plot_data([])
        else:
            super().__init__(Figure())

    def setup_chinese_font(self):
        try:
            import matplotlib
            matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
            matplotlib.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            logger.warning(f"设置中文字体失败: {str(e)}")

    def _draw_background_zones(self):
        """绘制稳定度背景色带 — 极简柔和色带（绿/黄/红三区）"""
        self.axes.axhspan(0, 4, alpha=0.06, color=_SUCCESS, zorder=0)
        self.axes.axhspan(4, 6, alpha=0.06, color=_WARNING, zorder=0)
        self.axes.axhspan(6, 10, alpha=0.06, color=_ERROR, zorder=0)

    def _draw_legend(self):
        """绘制图例 - 极简风格"""
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=_SUCCESS, alpha=0.3, label='稳定 (0-4)'),
            Patch(facecolor=_WARNING, alpha=0.3, label='波动 (4-6)'),
            Patch(facecolor=_ERROR, alpha=0.3, label='不稳定 (6-10)'),
        ]
        self.fig.legend(
            handles=legend_elements,
            loc='upper right',
            bbox_to_anchor=(0.98, 0.98),
            ncol=3,
            fontsize=8,
            frameon=False,
            columnspacing=1.2,
            handlelength=1.2,
            handletextpad=0.4
        )

    def plot_data(self, records):
        if not MATPLOTLIB_AVAILABLE:
            return

        try:
            self.axes.clear()
            # 清除旧的fig级别图例
            for leg in self.fig.legends:
                leg.remove()

            if not records:
                self.fig.set_facecolor(_BG)
                self.axes.set_facecolor(_BG)
                self.axes.set_title("情绪稳定度变化趋势", fontsize=13, color=_TEXT_PRIMARY, pad=12, loc='left')
                self.axes.set_ylabel("情绪稳定度 (0-10)", fontsize=10, color=_TEXT_SECONDARY, labelpad=10)
                self.axes.set_ylim(0, 10)
                self._apply_axes_style()
                self._draw_background_zones()
                self._draw_legend()
                self.fig.tight_layout(pad=2.0)
                self.fig.subplots_adjust(bottom=0.15, top=0.88)
                self.draw()
                return

            records_chronological = sorted(records, key=lambda x: x.get("timestamp", ""))
            n = len(records_chronological)
            x = list(range(1, n + 1))
            y = []
            timestamps = []
            for r in records_chronological:
                try:
                    score = r.get('情绪稳定度分数', r.get('anxiety_score', 0.0))
                    y.append(float(score))
                except (TypeError, ValueError):
                    y.append(0.0)
                timestamps.append(str(r.get('timestamp', '')))

            self.fig.set_facecolor(_BG)
            self.axes.set_facecolor(_BG)

            # 背景色带
            self._draw_background_zones()

            # 绘制趋势线 - 极简蓝色线条 + 圆形数据点
            self.axes.plot(x, y, color=_ACCENT, linewidth=2.0, marker='o',
                           markersize=5, markerfacecolor=_ACCENT,
                           markeredgecolor=_BG, markeredgewidth=1.2,
                           zorder=5)

            # 数据点标签（仅 <= 20 个时显示）
            if n <= 20:
                for xi, yi in zip(x, y):
                    self.axes.annotate(f'{yi:.1f}', (xi, yi), textcoords="offset points",
                                      xytext=(0, 8), ha='center', fontsize=8,
                                      color=_TEXT_SECONDARY, zorder=6)

            # 标题
            self.axes.set_title("情绪稳定度变化趋势", fontsize=14, color=_TEXT_PRIMARY, pad=15, loc='left')
            self.axes.set_ylabel("稳定度 (0–10)", fontsize=10, color=_TEXT_SECONDARY, labelpad=10)
            self.axes.set_ylim(-0.3, 10.3)
            self.axes.set_xlim(0.5, n + 0.5)
            self._apply_axes_style()

            # 设置 x 轴刻度
            if n <= 15:
                self.axes.set_xticks(x)
                date_labels = []
                for ts in timestamps:
                    try:
                        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                        date_labels.append(dt.strftime("%m/%d\n%H:%M"))
                    except (ValueError, TypeError):
                        date_labels.append(ts[-8:] if ts else "")
                self.axes.set_xticklabels(date_labels, rotation=0, ha='center', fontsize=7)
            elif n <= 40:
                step = max(1, n // 10)
                tick_positions = x[::step]
                tick_labels = []
                for i in range(0, n, step):
                    try:
                        dt = datetime.strptime(timestamps[i], "%Y-%m-%d %H:%M:%S")
                        tick_labels.append(dt.strftime("%m/%d %H:%M"))
                    except (ValueError, TypeError):
                        tick_labels.append("")
                self.axes.set_xticks(tick_positions)
                self.axes.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=7)
            else:
                tick_positions = [x[0], x[n//4], x[n//2], x[3*n//4], x[-1]]
                tick_labels = []
                for idx in [0, n//4, n//2, 3*n//4, n-1]:
                    try:
                        dt = datetime.strptime(timestamps[idx], "%Y-%m-%d %H:%M:%S")
                        tick_labels.append(dt.strftime("%m/%d %H:%M"))
                    except (ValueError, TypeError):
                        tick_labels.append("")
                self.axes.set_xticks(tick_positions)
                self.axes.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=7)

            self.axes.set_xlabel("", fontsize=10, labelpad=10)

            # 图例
            self._draw_legend()

            # 布局调整 — 增加底部空间防止 x 轴标签被截断
            self.fig.tight_layout(pad=2.5)
            self.fig.subplots_adjust(bottom=0.25, left=0.10, right=0.95, top=0.88)
            self.draw()
        except Exception as e:
            logger.error(f"绘图失败: {str(e)}", exc_info=True)

    def _apply_axes_style(self):
        """应用极简主义坐标轴样式"""
        # 隐藏顶/右边框
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        # 左/下边框改为细灰色
        self.axes.spines['left'].set_color(_BORDER)
        self.axes.spines['left'].set_linewidth(1)
        self.axes.spines['bottom'].set_color(_BORDER)
        self.axes.spines['bottom'].set_linewidth(1)
        # 刻度颜色
        self.axes.tick_params(colors=_TEXT_SECONDARY, length=3)
        # 细网格
        self.axes.grid(True, alpha=0.4, linestyle='-', linewidth=0.6, color=_BORDER, zorder=1)
