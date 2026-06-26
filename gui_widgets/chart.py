# -*- coding: utf-8 -*-
"""
趋势图表模块 — 构成主义风格

封装 matplotlib 的 FigureCanvas，用于在 PyQt5 界面中显示情绪稳定度趋势图。
支持分区域着色、中文字体、数据点标签和图例。

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


class MplCanvas(FigureCanvas):
    """
    matplotlib 图表画布类

    封装 matplotlib 的 FigureCanvas，用于在 PyQt5 界面中显示图表。
    主要功能：
    - 绘制情绪稳定度变化趋势图
    - 自动配置中文字体（微软雅黑/黑体）
    - 支持分区域着色（不同稳定度等级不同颜色）
    - 使用日期时间作为 x 轴标签
    - 数据点标签仅在数据点 <= 20 个时显示
    - 图例位于图表下方，不占用绘图区域

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
        """绘制稳定度背景色带 - 构成主义单色梯度"""
        self.axes.axhspan(0, 2, alpha=0.06, color='#2B2B2B', zorder=0)
        self.axes.axhspan(2, 4, alpha=0.04, color='#2B2B2B', zorder=0)
        self.axes.axhspan(4, 6, alpha=0.06, color='#8A8580', zorder=0)
        self.axes.axhspan(6, 8, alpha=0.08, color='#C44B4F', zorder=0)
        self.axes.axhspan(8, 10, alpha=0.12, color='#C44B4F', zorder=0)

    def _draw_legend(self):
        """绘制图例 - 构成主义简洁风格"""
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2B2B2B', alpha=0.3, label='稳定 (0-4)'),
            Patch(facecolor='#8A8580', alpha=0.4, label='波动 (4-6)'),
            Patch(facecolor='#C44B4F', alpha=0.5, label='不稳定 (6-10)'),
        ]
        self.fig.legend(
            handles=legend_elements,
            loc='upper right',
            bbox_to_anchor=(0.98, 0.98),
            ncol=3,
            fontsize=8,
            frameon=True,
            framealpha=0.9,
            edgecolor='#2B2B2B',
            borderpad=0.4,
            columnspacing=1.0,
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
                self.fig.set_facecolor('#F2EDE4')
                self.axes.set_facecolor('#F2EDE4')
                self.axes.set_title("情绪稳定度变化趋势", fontsize=13, fontweight='bold', pad=12, loc='left')
                self.axes.set_ylabel("情绪稳定度 (0-10)", fontsize=10, fontweight='bold', labelpad=10)
                self.axes.set_ylim(0, 10)
                self.axes.grid(True, alpha=0.3, linestyle='--', zorder=1)
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

            self.fig.set_facecolor('#F2EDE4')
            self.axes.set_facecolor('#F2EDE4')

            # 背景色带
            self._draw_background_zones()

            # 绘制趋势线 - 构成主义工业风格：砖红线条+方块数据点
            self.axes.plot(x, y, color='#C44B4F', linewidth=2, marker='s',
                           markersize=5, markerfacecolor='#C44B4F',
                           markeredgecolor='#2B2B2B', markeredgewidth=1,
                           zorder=5)

            # 数据点标签（仅 <= 20 个时显示）
            if n <= 20:
                for xi, yi in zip(x, y):
                    self.axes.annotate(f'{yi:.1f}', (xi, yi), textcoords="offset points",
                                      xytext=(0, 10), ha='center', fontsize=8,
                                      fontweight='bold', color='#2B2B2B', zorder=6)

            # x 轴标签：使用日期时间
            self.axes.set_title("情绪稳定度变化趋势", fontsize=13, fontweight='bold', pad=12, loc='left')
            self.axes.set_ylabel("情绪稳定度 (0-10)", fontsize=10, fontweight='bold', labelpad=10)
            self.axes.set_ylim(-0.3, 10.3)
            self.axes.set_xlim(0.5, n + 0.5)
            self.axes.grid(True, alpha=0.3, linestyle='--', zorder=1)

            # 设置 x 轴刻度
            if n <= 15:
                self.axes.set_xticks(x)
                date_labels = []
                for ts in timestamps:
                    try:
                        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                        date_labels.append(dt.strftime("%m-%d %H:%M"))
                    except (ValueError, TypeError):
                        date_labels.append(ts[-8:] if ts else "")
                self.axes.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=8)
            elif n <= 40:
                step = max(1, n // 10)
                tick_positions = x[::step]
                tick_labels = []
                for i in range(0, n, step):
                    try:
                        dt = datetime.strptime(timestamps[i], "%Y-%m-%d %H:%M:%S")
                        tick_labels.append(dt.strftime("%m-%d %H:%M"))
                    except (ValueError, TypeError):
                        tick_labels.append("")
                self.axes.set_xticks(tick_positions)
                self.axes.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
            else:
                tick_positions = [x[0], x[n//4], x[n//2], x[3*n//4], x[-1]]
                tick_labels = []
                for idx in [0, n//4, n//2, 3*n//4, n-1]:
                    try:
                        dt = datetime.strptime(timestamps[idx], "%Y-%m-%d %H:%M:%S")
                        tick_labels.append(dt.strftime("%m-%d %H:%M"))
                    except (ValueError, TypeError):
                        tick_labels.append("")
                self.axes.set_xticks(tick_positions)
                self.axes.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

            self.axes.set_xlabel("", fontsize=10, labelpad=10)

            # 图例
            self._draw_legend()

            # 布局调整
            self.fig.tight_layout(pad=2.5)
            self.fig.subplots_adjust(bottom=0.20, left=0.08, right=0.95, top=0.88)
            self.draw()
        except Exception as e:
            logger.error(f"绘图失败: {str(e)}", exc_info=True)
