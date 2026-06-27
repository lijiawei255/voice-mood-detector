# -*- coding: utf-8 -*-
"""
科研级评估扩展卡片组件

提供 P1/P2 新增指标的可视化展示：
- 声学特征卡片（F0、jitter、shimmer、HNR、语速、静音比）
- 心理状态指标卡片（压力、焦虑、低落、激活、语音稳定性）
- 可靠性徽章（综合可靠性 + 双模型一致性）
- 基线偏移卡片（相对个人基线的偏移）

设计风格：苏联构成主义（粗炭黑边框、直角、三角装饰、砖红点缀）

作者：Jiawei Li
许可证：GPL v3
"""

from PyQt5.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout, QSizePolicy, QWidget
)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QPolygon


class _ConstructivistCard(QFrame):
    """构成主义风格卡片基类"""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("constructivistCard")
        self._title = title
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setStyleSheet("""
            QFrame#constructivistCard {
                background-color: #F2EDE4;
                border: 3px solid #2B2B2B;
                border-radius: 0px;
            }
        """)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        # 粗黑边框
        pen = QPen(QColor("#2B2B2B"))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(1, 1, self.width() - 3, self.height() - 3)
        # 左上角红色楔形
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#C44B4F"))
        wedge = QPolygon([QPoint(0, 0), QPoint(22, 0), QPoint(0, 22)])
        painter.drawPolygon(wedge)
        # 右下角黑色小三角
        painter.setBrush(QColor("#2B2B2B"))
        w, h = self.width(), self.height()
        tri = QPolygon([QPoint(w, h), QPoint(w - 12, h), QPoint(w, h - 12)])
        painter.drawPolygon(tri)
        painter.end()
        super().paintEvent(event)


class MetricRow(QWidget):
    """一行指标：标签 + 数值 + 可选单位"""

    def __init__(self, label, value="--", unit="", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        self.label_lbl = QLabel(label)
        self.label_lbl.setFont(QFont("Microsoft YaHei", 10))
        self.label_lbl.setStyleSheet("color: #8A8580;")
        layout.addWidget(self.label_lbl)

        layout.addStretch()

        self.value_lbl = QLabel(value)
        self.value_lbl.setFont(QFont("Consolas", 11, QFont.Bold))
        self.value_lbl.setStyleSheet("color: #2B2B2B;")
        layout.addWidget(self.value_lbl)

        if unit:
            self.unit_lbl = QLabel(unit)
            self.unit_lbl.setFont(QFont("Microsoft YaHei", 9))
            self.unit_lbl.setStyleSheet("color: #8A8580;")
            layout.addWidget(self.unit_lbl)
        else:
            self.unit_lbl = None

    def set_value(self, text):
        self.value_lbl.setText(str(text))


class AcousticFeatureCard(_ConstructivistCard):
    """声学特征卡片"""

    def __init__(self, parent=None):
        super().__init__("声学特征", parent)
        self._setup_ui()
        self.hide()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(8)

        title = QLabel("▸ 声学特征")
        title.setFont(QFont("Microsoft YaHei", 12, QFont.Black))
        title.setStyleSheet("color: #C44B4F; font-weight: 900;")
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

        note = QLabel("⚠ 声带特征需要安装 praat-parselmouth 以获得专业级精度")
        note.setFont(QFont("Microsoft YaHei", 9))
        note.setStyleSheet("color: #8A8580;")
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


class PsychologicalIndicatorCard(_ConstructivistCard):
    """心理状态指标卡片"""

    def __init__(self, parent=None):
        super().__init__("心理状态指标", parent)
        self._setup_ui()
        self.hide()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(8)

        title = QLabel("▸ 心理状态指标")
        title.setFont(QFont("Microsoft YaHei", 12, QFont.Black))
        title.setStyleSheet("color: #C44B4F; font-weight: 900;")
        layout.addWidget(title)

        self.stress_row = MetricRow("压力指数", "--")
        self.anxiety_row = MetricRow("焦虑指数", "--")
        self.depression_row = MetricRow("低落倾向", "--")
        self.activation_row = MetricRow("情绪激活度", "--")
        self.speech_stability_row = MetricRow("语音稳定性", "--")

        for row in [self.stress_row, self.anxiety_row, self.depression_row,
                    self.activation_row, self.speech_stability_row]:
            layout.addWidget(row)

        note = QLabel("⚠ 基于声学特征与 VAD 维度的估计值，非临床诊断")
        note.setFont(QFont("Microsoft YaHei", 9))
        note.setStyleSheet("color: #8A8580;")
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


class ReliabilityBadge(_ConstructivistCard):
    """可靠性徽章卡片"""

    def __init__(self, parent=None):
        super().__init__("可靠性", parent)
        self._setup_ui()
        self.hide()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(8)

        title = QLabel("▸ 可靠性评估")
        title.setFont(QFont("Microsoft YaHei", 12, QFont.Black))
        title.setStyleSheet("color: #C44B4F; font-weight: 900;")
        layout.addWidget(title)

        self.level_lbl = QLabel("--")
        self.level_lbl.setFont(QFont("Consolas", 24, QFont.Bold))
        self.level_lbl.setAlignment(Qt.AlignCenter)
        self.level_lbl.setMinimumHeight(40)
        layout.addWidget(self.level_lbl)

        self.detail_lbl = QLabel("")
        self.detail_lbl.setFont(QFont("Microsoft YaHei", 10))
        self.detail_lbl.setAlignment(Qt.AlignCenter)
        self.detail_lbl.setWordWrap(True)
        self.detail_lbl.setStyleSheet("color: #8A8580;")
        layout.addWidget(self.detail_lbl)

    def update_reliability(self, result):
        reliability = result.get('assessment_reliability', '') or result.get('result_reliability', '')
        if not reliability:
            self.hide()
            return

        colors = {"高": "#C44B4F", "中": "#8A8580", "低": "#C44B4F"}
        color = colors.get(reliability, "#8A8580")
        self.level_lbl.setText(f"{reliability}")
        self.level_lbl.setStyleSheet(f"color: {color}; font-weight: bold;")

        details = []
        if result.get('is_dual_model') or result.get('is_research_session'):
            agreement = result.get('model_agreement')
            if agreement is not None:
                details.append(f"模型一致性: {agreement:.0%}")
            if result.get('result_reliability'):
                details.append(f"规模敏感性: {result['result_reliability']}")
        if result.get('consistency') is not None:
            details.append(f"多次采样一致性: {result['consistency']:.2f}")

        self.detail_lbl.setText(" | ".join(details) if details else "基于音频质量与模型置信度")
        self.show()


class BaselineDeviationCard(_ConstructivistCard):
    """基线偏移卡片"""

    def __init__(self, parent=None):
        super().__init__("基线偏移", parent)
        self._setup_ui()
        self.hide()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(8)

        title = QLabel("▸ 相对个人基线")
        title.setFont(QFont("Microsoft YaHei", 12, QFont.Black))
        title.setStyleSheet("color: #C44B4F; font-weight: 900;")
        layout.addWidget(title)

        self.status_lbl = QLabel("基线尚未建立")
        self.status_lbl.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.status_lbl.setStyleSheet("color: #8A8580;")
        layout.addWidget(self.status_lbl)

        self.summary_lbl = QLabel("")
        self.summary_lbl.setFont(QFont("Microsoft YaHei", 10))
        self.summary_lbl.setWordWrap(True)
        self.summary_lbl.setStyleSheet("color: #2B2B2B;")
        layout.addWidget(self.summary_lbl)

    def update_deviation(self, deviation):
        if not deviation or not deviation.get('available'):
            self.status_lbl.setText("基线尚未建立")
            self.status_lbl.setStyleSheet("color: #8A8580;")
            self.summary_lbl.setText("建议在「基线校准」面板采集 3-5 条平静语音")
            self.show()
            return

        level = deviation.get('personalized_stability_level', '与基线一致')
        score = deviation.get('personalized_stability_score', 0.0)
        self.status_lbl.setText(f"{level}（评分 {score:.1f}）")
        self.status_lbl.setStyleSheet("color: #C44B4F; font-weight: bold;")
        self.summary_lbl.setText(deviation.get('summary', ''))
        self.show()
