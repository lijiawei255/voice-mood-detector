# -*- coding: utf-8 -*-
"""
统计分析面板 (P2)

提供历史数据的统计可视化和分析结果展示。
极简主义风格。

作者：Jiawei Li
许可证：GPL v3
"""

from PyQt5.QtWidgets import (
    QVBoxLayout, QGroupBox, QTextEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from .fonts import UI_FONT, MONO_FONT


# 极简主义色板常量
_TEXT_PRIMARY = "#1D1D1F"
_TEXT_SECONDARY = "#86868B"
_BORDER = "#D2D2D7"
_BG_SECONDARY = "#F5F5F7"


class StatsPanel(QGroupBox):
    """统计分析面板 — 极简主义风格"""

    def __init__(self, parent=None):
        super().__init__("统计分析", parent)
        self.setFont(QFont(UI_FONT, 12, QFont.Medium))
        self.setObjectName("statsGroup")
        self.setStyleSheet(f"""
            QGroupBox#statsGroup {{
                border: 1px solid {_BORDER};
                border-radius: 8px;
                margin-top: 16px;
                padding-top: 18px;
                background-color: #FFFFFF;
            }}
            QGroupBox#statsGroup::title {{
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
                color: {_TEXT_PRIMARY};
            }}
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self.content = QTextEdit()
        self.content.setReadOnly(True)
        self.content.setFont(QFont(MONO_FONT, 11))
        self.content.setMinimumHeight(200)
        self.content.setStyleSheet(f"""
            QTextEdit {{
                border: 1px solid {_BORDER};
                border-radius: 6px;
                background-color: {_BG_SECONDARY};
                padding: 12px;
                color: {_TEXT_PRIMARY};
            }}
        """)
        layout.addWidget(self.content)

    def update_stats(self, history_manager):
        """根据历史数据更新统计显示"""
        if history_manager is None:
            self.content.setPlainText("暂无数据")
            return

        try:
            stats = history_manager.get_statistics()
            stability = stats.get("stability", {})
            valence = stats.get("valence", {})
            arousal = stats.get("arousal", {})
            dominance = stats.get("dominance", {})
            emo_dist = stats.get("emotion_distribution", {})
            compound_dist = stats.get("compound_distribution", {})

            lines = []
            lines.append("─" * 44)
            lines.append(f"  记录总数: {stats.get('record_count', 0)}")
            lines.append("")

            if stability:
                lines.append("  情绪稳定度")
                lines.append(f"    均值: {stability.get('mean', 0):.2f}  "
                           f"标准差: {stability.get('std', 0):.2f}")
                lines.append(f"    范围: {stability.get('min', 0):.1f} - "
                           f"{stability.get('max', 0):.1f}")
                lines.append("")

            if valence or arousal:
                lines.append("  VAD 维度均值")
                lines.append(f"    效价: {valence.get('mean', 0):+.3f}   "
                           f"唤醒度: {arousal.get('mean', 0):.3f}   "
                           f"掌控感: {dominance.get('mean', 0):.3f}")
                lines.append("")

            if emo_dist:
                lines.append("  情绪分布")
                sorted_emos = sorted(emo_dist.items(), key=lambda x: -x[1])
                for emo, count in sorted_emos:
                    pct = count / stats["record_count"] * 100
                    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                    lines.append(f"    {emo:4s} {bar} {count}次 ({pct:.0f}%)")
                lines.append("")

            if compound_dist:
                lines.append("  复合情绪分布")
                sorted_comp = sorted(compound_dist.items(), key=lambda x: -x[1])
                for comp, count in sorted_comp:
                    lines.append(f"    {comp}: {count}次")
                lines.append("")

            # 趋势
            stability_summary = history_manager.get_stability_summary()
            if stability_summary:
                lines.append("  趋势")
                lines.append(f"    稳定度趋势: {stability_summary.get('trend', 'N/A')}")
                lines.append(f"    近期均值: {stability_summary.get('recent_mean', 0):.2f}")

            lines.append("─" * 44)
            self.content.setPlainText("\n".join(lines))

        except Exception as e:
            self.content.setPlainText(f"统计计算失败: {e}")
