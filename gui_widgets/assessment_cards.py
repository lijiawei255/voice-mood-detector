# -*- coding: utf-8 -*-
"""
科研级评估扩展卡片组件

提供 P1/P2 新增指标的可视化展示：
- 声学特征卡片（F0、jitter、shimmer、HNR、语速、静音比）
- 心理状态指标卡片（压力、焦虑、低落、激活、语音稳定性）
- 可靠性徽章（综合可靠性 + 双模型一致性）
- 基线偏移卡片（相对个人基线的偏移）

设计风格：极简主义（瑞士/包豪斯）— 细线边框、浅灰底、蓝色强调、无装饰

作者：Jiawei Li
许可证：GPL v3
"""

from PyQt5.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, QWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from .fonts import UI_FONT, MONO_FONT


# 极简主义色板常量
_BG_SECONDARY = "#F5F5F7"
_TEXT_PRIMARY = "#1D1D1F"
_TEXT_SECONDARY = "#86868B"
_ACCENT = "#1A73E8"
_BORDER = "#D2D2D7"
_SUCCESS = "#34A853"
_WARNING = "#F9AB00"
_ERROR = "#EA4335"


class _MinimalCard(QFrame):
    """极简主义风格卡片基类 — 浅灰背景 + 细线边框 + 圆角，无装饰绘制"""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("minimalCard")
        self._title = title
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setStyleSheet(f"""
            QFrame#minimalCard {{
                background-color: {_BG_SECONDARY};
                border: 1px solid {_BORDER};
                border-radius: 8px;
            }}
        """)


class MetricRow(QWidget):
    """一行指标：标签 + 数值 + 可选单位（极简风格）"""

    def __init__(self, label, value="--", unit="", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(8)

        self.label_lbl = QLabel(label)
        self.label_lbl.setFont(QFont(UI_FONT, 11))
        self.label_lbl.setStyleSheet(f"color: {_TEXT_SECONDARY}; background: transparent;")
        layout.addWidget(self.label_lbl)

        layout.addStretch()

        self.value_lbl = QLabel(value)
        self.value_lbl.setFont(QFont(MONO_FONT, 12, QFont.Bold))
        self.value_lbl.setStyleSheet(f"color: {_TEXT_PRIMARY}; background: transparent;")
        layout.addWidget(self.value_lbl)

        if unit:
            self.unit_lbl = QLabel(unit)
            self.unit_lbl.setFont(QFont(UI_FONT, 10))
            self.unit_lbl.setStyleSheet(f"color: {_TEXT_SECONDARY}; background: transparent;")
            layout.addWidget(self.unit_lbl)
        else:
            self.unit_lbl = None

    def set_value(self, text):
        self.value_lbl.setText(str(text))


class AcousticFeatureCard(_MinimalCard):
    """声学特征卡片"""

    def __init__(self, parent=None):
        super().__init__("声学特征", parent)
        self._setup_ui()
        self.hide()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        title = QLabel("声学特征")
        title.setFont(QFont(UI_FONT, 13, QFont.Medium))
        title.setStyleSheet(f"color: {_TEXT_PRIMARY}; background: transparent;")
        layout.addWidget(title)

        self.f0_row = MetricRow("平均音高 F0", "--", "Hz")
        self.f0_std_row = MetricRow("F0 标准差", "--", "Hz")
        self.jitter_row = MetricRow("Jitter", "--", "%")
        self.shimmer_row = MetricRow("Shimmer", "--", "%")
        self.hnr_row = MetricRow("HNR", "--", "dB")
        self.rate_row = MetricRow("语速", "--", "音节/秒")
        self.silence_row = MetricRow("静音比例", "--", "%")

        for row in [self.f0_row, self.f0_std_row, self.jitter_row,
                    self.shimmer_row, self.hnr_row, self.rate_row, self.silence_row]:
            layout.addWidget(row)

        note = QLabel("声带音质指标（Jitter / Shimmer / HNR）默认由 praat-parselmouth（Praat 算法，临床语音分析金标准）提取；未安装时将自动降级为 librosa 近似，精度较低。")
        note.setFont(QFont(UI_FONT, 10))
        note.setStyleSheet(f"color: {_TEXT_SECONDARY}; background: transparent;")
        note.setWordWrap(True)
        layout.addWidget(note)

    def update_features(self, features):
        if not features:
            self.hide()
            return
        self.f0_row.set_value(f"{features.get('f0_mean', 0):.1f}")
        self.f0_std_row.set_value(f"{features.get('f0_std', 0):.1f}")
        self.jitter_row.set_value(f"{features.get('jitter_local', 0) * 100:.2f}")
        self.shimmer_row.set_value(f"{features.get('shimmer_local', 0) * 100:.2f}")
        self.hnr_row.set_value(f"{features.get('hnr_mean', 0):.1f}")
        self.rate_row.set_value(f"{features.get('speech_rate', 0):.2f}")
        self.silence_row.set_value(f"{features.get('silence_ratio', 0) * 100:.1f}")
        self.show()


class PsychologicalIndicatorCard(_MinimalCard):
    """心理状态指标卡片"""

    def __init__(self, parent=None):
        super().__init__("心理状态指标", parent)
        self._setup_ui()
        self.hide()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        title = QLabel("心理状态指标")
        title.setFont(QFont(UI_FONT, 13, QFont.Medium))
        title.setStyleSheet(f"color: {_TEXT_PRIMARY}; background: transparent;")
        layout.addWidget(title)

        self.stress_row = MetricRow("压力指数", "--")
        self.anxiety_row = MetricRow("焦虑指数", "--")
        self.depression_row = MetricRow("低落倾向", "--")
        self.activation_row = MetricRow("情绪激活度", "--")
        self.speech_stability_row = MetricRow("语音稳定性", "--")

        for row in [self.stress_row, self.anxiety_row, self.depression_row,
                    self.activation_row, self.speech_stability_row]:
            layout.addWidget(row)

        note = QLabel("基于声学特征与 VAD 维度的估计值，非临床诊断")
        note.setFont(QFont(UI_FONT, 10))
        note.setStyleSheet(f"color: {_TEXT_SECONDARY}; background: transparent;")
        note.setWordWrap(True)
        layout.addWidget(note)

    def update_indicators(self, indicators):
        if not indicators:
            self.hide()
            return
        self.stress_row.set_value(f"{indicators.get('stress_index', 0):.2f}")
        self.anxiety_row.set_value(f"{indicators.get('anxiety_index', 0):.2f}")
        self.depression_row.set_value(f"{indicators.get('depression_tendency_index', 0):.2f}")
        self.activation_row.set_value(f"{indicators.get('emotional_activation', 0):.2f}")
        self.speech_stability_row.set_value(f"{indicators.get('speech_stability', 0):.2f}")
        self.show()


class ReliabilityBadge(_MinimalCard):
    """可靠性徽章卡片"""

    def __init__(self, parent=None):
        super().__init__("可靠性", parent)
        self._setup_ui()
        self.hide()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        title = QLabel("可靠性评估")
        title.setFont(QFont(UI_FONT, 13, QFont.Medium))
        title.setStyleSheet(f"color: {_TEXT_PRIMARY}; background: transparent;")
        layout.addWidget(title)

        self.level_lbl = QLabel("--")
        self.level_lbl.setFont(QFont(MONO_FONT, 28, QFont.Bold))
        self.level_lbl.setAlignment(Qt.AlignCenter)
        self.level_lbl.setMinimumHeight(40)
        self.level_lbl.setStyleSheet(f"color: {_TEXT_PRIMARY}; background: transparent;")
        layout.addWidget(self.level_lbl)

        self.detail_lbl = QLabel("")
        self.detail_lbl.setFont(QFont(UI_FONT, 11))
        self.detail_lbl.setAlignment(Qt.AlignCenter)
        self.detail_lbl.setWordWrap(True)
        self.detail_lbl.setStyleSheet(f"color: {_TEXT_SECONDARY}; background: transparent;")
        layout.addWidget(self.detail_lbl)

    def update_reliability(self, result):
        reliability = result.get('assessment_reliability', '') or result.get('result_reliability', '')
        if not reliability:
            self.hide()
            return

        # 极简色彩编码：高=绿、中=灰、低=红
        colors = {"高": _SUCCESS, "中": _TEXT_SECONDARY, "低": _ERROR}
        color = colors.get(reliability, _TEXT_SECONDARY)
        self.level_lbl.setText(f"{reliability}")
        self.level_lbl.setStyleSheet(f"color: {color}; font-weight: bold; background: transparent;")

        details = []
        if result.get('is_dual_model') or result.get('is_research_session'):
            agreement = result.get('model_agreement')
            if agreement is not None:
                details.append(f"模型一致性: {agreement:.0%}")
            if result.get('result_reliability'):
                details.append(f"规模敏感性: {result['result_reliability']}")
        if result.get('consistency') is not None:
            details.append(f"多次采样一致性: {result['consistency']:.2f}")

        self.detail_lbl.setText("  ·  ".join(details) if details else "基于音频质量与模型置信度")
        self.show()


class BaselineDeviationCard(_MinimalCard):
    """基线偏移卡片"""

    def __init__(self, parent=None):
        super().__init__("基线偏移", parent)
        self._setup_ui()
        self.hide()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        title = QLabel("相对个人基线")
        title.setFont(QFont(UI_FONT, 13, QFont.Medium))
        title.setStyleSheet(f"color: {_TEXT_PRIMARY}; background: transparent;")
        layout.addWidget(title)

        self.status_lbl = QLabel("基线尚未建立")
        self.status_lbl.setFont(QFont(UI_FONT, 12))
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.status_lbl.setStyleSheet(f"color: {_TEXT_SECONDARY}; background: transparent;")
        layout.addWidget(self.status_lbl)

        self.summary_lbl = QLabel("")
        self.summary_lbl.setFont(QFont(UI_FONT, 11))
        self.summary_lbl.setWordWrap(True)
        self.summary_lbl.setStyleSheet(f"color: {_TEXT_PRIMARY}; background: transparent;")
        layout.addWidget(self.summary_lbl)

    def update_deviation(self, deviation):
        if not deviation or not deviation.get('available'):
            self.status_lbl.setText("基线尚未建立")
            self.status_lbl.setStyleSheet(f"color: {_TEXT_SECONDARY}; background: transparent;")
            self.summary_lbl.setText("建议在「基线校准」面板采集 3-5 条平静语音")
            self.show()
            return

        level = deviation.get('personalized_stability_level', '与基线一致')
        score = deviation.get('personalized_stability_score', 0.0)
        self.status_lbl.setText(f"{level}（评分 {score:.1f}）")
        self.status_lbl.setStyleSheet(f"color: {_ACCENT}; font-weight: bold; background: transparent;")
        self.summary_lbl.setText(deviation.get('summary', ''))
        self.show()
