# -*- coding: utf-8 -*-
"""
科研级评估报告卡片组件

提供结构化的情绪评估结果展示，包含：
- 主要情绪卡片（含置信度）
- 稳定度卡片（含三因子分项）
- VAD 维度指示条（效价/唤醒度/掌控感）
- 音频质量反馈指示
- 可靠性标签

设计风格：极简主义（瑞士/包豪斯）— 细线边框、浅灰底、蓝色强调、无装饰

作者：Jiawei Li
许可证：GPL v3
"""

from PyQt5.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QSizePolicy, QWidget
)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QFont, QColor, QPainter, QBrush, QFontMetrics

from .fonts import UI_FONT, MONO_FONT


# 极简主义色板常量（与 gui.py 主样式保持一致）
_BG_SECONDARY = "#F5F5F7"
_BG_TERTIARY = "#EBEBEF"
_TEXT_PRIMARY = "#1D1D1F"
_TEXT_SECONDARY = "#86868B"
_ACCENT = "#1A73E8"
_BORDER = "#D2D2D7"
_SUCCESS = "#34A853"
_WARNING = "#F9AB00"
_ERROR = "#EA4335"


class DimensionBar(QWidget):
    """
    VAD 维度指示条

    用于可视化展示效价(Valence)、唤醒度(Arousal)、掌控感(Dominance)三个维度。
    每个维度显示为一个水平条形指示器，左侧标签 + 彩色条 + 数值。

    极简主义风格：细圆角条、中性背景、蓝色强调。
    """

    def __init__(self, label, value, range_min, range_max,
                 low_color=_ACCENT, high_color=_ACCENT, parent=None):
        """
        参数：
            label (str): 维度名称
            value (float): 当前值
            range_min (float): 最小值
            range_max (float): 最大值
            low_color (str): 低值端颜色
            high_color (str): 高值端颜色
        """
        super().__init__(parent)
        self._label = label
        self._value = value
        self._range_min = range_min
        self._range_max = range_max
        self._low_color = QColor(low_color)
        self._high_color = QColor(high_color)
        # 用于绘制文本的字体（与 paintEvent 保持一致）
        self._label_font = QFont(UI_FONT, 11)
        self._value_font = QFont(MONO_FONT, 11, QFont.Bold)
        # 根据字体度量动态计算标签与数值列宽，保证不截断
        fm_label = QFontMetrics(self._label_font)
        fm_value = QFontMetrics(self._value_font)
        self._label_w = max(72, fm_label.width(self._label) + 12)
        self._value_w = max(60, fm_value.width("-0.00") + 12)
        self.setMinimumHeight(32)
        self.setMaximumHeight(40)
        # 保证标签 + 条形最小宽度 + 数值 不致 bar_w 变为负值
        self.setMinimumWidth(self._label_w + self._value_w + 40)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w = self.width()
        h = self.height()

        # 标签
        painter.setFont(self._label_font)
        painter.setPen(QColor(_TEXT_SECONDARY))
        label_rect = QRect(0, 0, self._label_w, h)
        painter.drawText(label_rect, Qt.AlignLeft | Qt.AlignVCenter, self._label)

        # 条形背景（圆角）
        bar_x = self._label_w + 6
        bar_w = max(0, w - bar_x - self._value_w - 4)
        bar_h = 8
        bar_y = (h - bar_h) // 2
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(_BG_TERTIARY))
        painter.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 4, 4)

        # 计算填充比例（居中于零点或从最小值开始）
        if self._range_min < 0:
            # 双端条形（如效价，有正负）
            zero_pos = bar_x + int(bar_w * abs(self._range_min) /
                                   (self._range_max - self._range_min))
            if self._value >= 0:
                fill_w = int(bar_w * self._value / (self._range_max - self._range_min))
                fill_x = zero_pos
                fill_color = self._high_color
            else:
                fill_w = int(bar_w * abs(self._value) / (self._range_max - self._range_min))
                fill_x = zero_pos - fill_w
                fill_color = self._low_color
            # 画零点标记线
            painter.setPen(QColor(_TEXT_SECONDARY))
            painter.drawLine(zero_pos, bar_y - 3, zero_pos, bar_y + bar_h + 3)
            painter.setPen(Qt.NoPen)
        else:
            # 单端条形（如唤醒度，0~1）
            fill_ratio = max(0.0, min(1.0, (self._value - self._range_min) /
                               (self._range_max - self._range_min)))
            fill_w = int(bar_w * fill_ratio)
            fill_x = bar_x
            # 颜色插值
            fill_color = QColor(
                int(self._low_color.red() + (self._high_color.red() - self._low_color.red()) * fill_ratio),
                int(self._low_color.green() + (self._high_color.green() - self._low_color.green()) * fill_ratio),
                int(self._low_color.blue() + (self._high_color.blue() - self._low_color.blue()) * fill_ratio)
            )

        painter.setBrush(QBrush(fill_color))
        painter.drawRoundedRect(fill_x, bar_y, fill_w, bar_h, 4, 4)

        # 数值文本
        painter.setPen(QColor(_TEXT_PRIMARY))
        painter.setFont(self._value_font)
        val_str = f"{self._value:+.2f}" if self._range_min < 0 else f"{self._value:.2f}"
        val_rect = QRect(bar_x + bar_w + 6, 0, self._value_w, h)
        painter.drawText(val_rect, Qt.AlignLeft | Qt.AlignVCenter, val_str)

        painter.end()


class ResultCardWidget(QFrame):
    """
    集成化评估结果卡片部件

    在一个卡片中展示：
    - 主要情绪 + 置信度
    - VAD 维度（效价/唤醒度/掌控感）
    - 稳定度 + 三因子分项
    - 音频质量反馈
    - 可靠性标签

    设计风格：极简主义
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("resultCardWidget")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumHeight(300)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        # --- 第一行：主要情绪 + 稳定度 ---
        top_row = QHBoxLayout()
        top_row.setSpacing(16)

        # 主要情绪
        self.emotion_frame = self._make_sub_card("主要情绪")
        emotion_layout = QVBoxLayout(self.emotion_frame)
        self.emotion_value = QLabel("--")
        self.emotion_value.setFont(QFont(MONO_FONT, 36, QFont.Bold))
        self.emotion_value.setAlignment(Qt.AlignCenter)
        self.emotion_value.setStyleSheet(f"color: {_TEXT_PRIMARY}; background: transparent;")
        self.emotion_confidence = QLabel("")
        self.emotion_confidence.setFont(QFont(UI_FONT, 11))
        self.emotion_confidence.setAlignment(Qt.AlignCenter)
        self.emotion_confidence.setStyleSheet(f"color: {_TEXT_SECONDARY}; background: transparent;")
        emotion_layout.addWidget(self.emotion_value)
        emotion_layout.addWidget(self.emotion_confidence)
        top_row.addWidget(self.emotion_frame, 1)

        # 稳定度
        self.stability_frame = self._make_sub_card("情绪稳定度 (0-10)")
        stability_layout = QVBoxLayout(self.stability_frame)
        self.stability_value = QLabel("--")
        self.stability_value.setFont(QFont(MONO_FONT, 36, QFont.Bold))
        self.stability_value.setAlignment(Qt.AlignCenter)
        self.stability_value.setStyleSheet(f"color: {_TEXT_PRIMARY}; background: transparent;")
        self.stability_level = QLabel("")
        self.stability_level.setFont(QFont(UI_FONT, 11))
        self.stability_level.setAlignment(Qt.AlignCenter)
        self.stability_level.setStyleSheet(f"color: {_TEXT_SECONDARY}; background: transparent;")
        stability_layout.addWidget(self.stability_value)
        stability_layout.addWidget(self.stability_level)
        top_row.addWidget(self.stability_frame, 1)

        layout.addLayout(top_row)

        # --- 第二行：VAD 维度 ---
        vad_label = QLabel("情感维度指标（从离散概率估计）")
        vad_label.setFont(QFont(UI_FONT, 11))
        vad_label.setStyleSheet(f"color: {_TEXT_SECONDARY};")
        layout.addWidget(vad_label)

        self.valence_bar = DimensionBar("效价", 0.0, -1.0, 1.0,
                                        low_color=_ACCENT, high_color=_ACCENT)
        self.arousal_bar = DimensionBar("唤醒度", 0.0, 0.0, 1.0,
                                        low_color=_TEXT_SECONDARY, high_color=_ACCENT)
        self.dominance_bar = DimensionBar("掌控感", 0.0, 0.0, 1.0,
                                          low_color=_TEXT_SECONDARY, high_color=_ACCENT)

        layout.addWidget(self.valence_bar)
        layout.addWidget(self.arousal_bar)
        layout.addWidget(self.dominance_bar)

        # --- 第三行：稳定度分项 ---
        factors_label = QLabel("稳定度因子分解（权重来源：专家设定）")
        factors_label.setFont(QFont(UI_FONT, 11))
        factors_label.setStyleSheet(f"color: {_TEXT_SECONDARY};")
        layout.addWidget(factors_label)

        factors_row = QHBoxLayout()
        factors_row.setSpacing(8)
        for name, key in [("负面加权", "negative"), ("熵分散度", "entropy"), ("极端度", "extremity")]:
            fframe = self._make_sub_card(name)
            flay = QVBoxLayout(fframe)
            flay.setContentsMargins(8, 8, 8, 8)
            flay.setSpacing(4)
            # 因子名称标签
            name_lbl = QLabel(name)
            name_lbl.setFont(QFont(UI_FONT, 11))
            name_lbl.setAlignment(Qt.AlignCenter)
            name_lbl.setStyleSheet(f"color: {_TEXT_SECONDARY}; background: transparent;")
            flay.addWidget(name_lbl)
            # 因子数值
            val = QLabel("--")
            val.setFont(QFont(MONO_FONT, 18, QFont.Bold))
            val.setAlignment(Qt.AlignCenter)
            val.setStyleSheet(f"color: {_TEXT_PRIMARY}; background: transparent;")
            flay.addWidget(val)
            setattr(self, f"factor_{key}_value", val)
            # 添加tooltip说明
            tooltips = {
                "负面加权": "负性情绪占比越高，分数越大",
                "熵分散度": "情绪分布越分散，分数越大",
                "极端度": "单类概率越极端，分数越大"
            }
            fframe.setToolTip(tooltips.get(name, ""))
            factors_row.addWidget(fframe)
        layout.addLayout(factors_row)

        # --- 音频质量标签 ---
        self.quality_label = QLabel("")
        self.quality_label.setFont(QFont(UI_FONT, 10))
        self.quality_label.setStyleSheet(f"color: {_TEXT_SECONDARY}; padding: 2px 0;")
        self.quality_label.hide()
        layout.addWidget(self.quality_label)

        # --- 可靠性标签 ---
        self.reliability_label = QLabel("")
        self.reliability_label.setFont(QFont(UI_FONT, 11))
        self.reliability_label.setAlignment(Qt.AlignCenter)
        self.reliability_label.setMinimumHeight(32)
        self.reliability_label.hide()
        layout.addWidget(self.reliability_label)

    def _make_sub_card(self, title):
        """创建一个子卡片框架（极简风格：浅灰背景 + 细线边框 + 圆角）"""
        frame = QFrame()
        frame.setObjectName("subCard")
        frame.setFrameStyle(QFrame.NoFrame)
        frame.setStyleSheet(f"""
            QFrame#subCard {{
                background-color: {_BG_SECONDARY};
                border: 1px solid {_BORDER};
                border-radius: 4px;
            }}
        """)
        return frame

    def update_result(self, result):
        """
        根据情绪识别结果更新卡片显示

        参数：
            result (dict): EmotionRecognizer.predict() 返回的结果字典
        """
        if not isinstance(result, dict) or not result.get('success', False):
            self._clear()
            return

        score = result.get('情绪稳定度分数', 0.0)
        level = result.get('情绪状态等级', '未知')
        color = result.get('等级颜色', _TEXT_SECONDARY)
        main_emotion = result.get('主要情绪', '未知')
        confidence = result.get('置信度', 0.0)

        # 主要情绪
        self.emotion_value.setText(main_emotion)
        self.emotion_value.setStyleSheet(f"color: {_TEXT_PRIMARY}; background: transparent;")
        self.emotion_confidence.setText(f"置信度 {confidence:.1%}")

        # 稳定度
        self.stability_value.setText(f"{score:.1f}")
        self.stability_value.setStyleSheet(f"color: {color}; background: transparent;")
        self.stability_level.setText(level)
        self.stability_level.setStyleSheet(f"color: {color}; background: transparent;")

        # VAD 维度
        valence = result.get('valence_score', 0.0)
        arousal = result.get('arousal_score', 0.0)
        dominance = result.get('dominance_score', 0.0)

        self.valence_bar._value = valence
        self.arousal_bar._value = arousal
        self.dominance_bar._value = dominance
        self.valence_bar.update()
        self.arousal_bar.update()
        self.dominance_bar.update()

        # 稳定度分项
        factors = result.get('稳定度分项', {})
        neg_score = factors.get('negative_weight_score', 0.0)
        ent_score = factors.get('entropy_score', 0.0)
        ext_score = factors.get('extremity_score', 0.0)

        self.factor_negative_value.setText(f"{neg_score:.1f}")
        self.factor_entropy_value.setText(f"{ent_score:.1f}")
        self.factor_extremity_value.setText(f"{ext_score:.1f}")

        # 音频质量（如果有）
        audio_quality = result.get('audio_quality', None)
        if audio_quality and isinstance(audio_quality, dict):
            q_score = audio_quality.get('quality_score', 0.0)
            q_label = audio_quality.get('quality_label', '')
            issues = audio_quality.get('issues', [])
            if issues:
                self.quality_label.setText(f"录音质量: {q_label} ({q_score:.2f})  ·  {', '.join(issues[:2])}")
            else:
                self.quality_label.setText(f"录音质量: {q_label} ({q_score:.2f})")
            self.quality_label.show()
        else:
            self.quality_label.hide()

        # 可靠性标签 — 极简色彩编码（高=绿、中=灰、低=红）
        reliability = result.get('assessment_reliability', '')
        if reliability:
            rel_colors = {"高": _SUCCESS, "中": _TEXT_SECONDARY, "低": _ERROR}
            rel_color = rel_colors.get(reliability, _TEXT_SECONDARY)
            self.reliability_label.setText(f"可靠性: {reliability}")
            self.reliability_label.setStyleSheet(
                f"color: {rel_color}; font-weight: bold; background: {_BG_SECONDARY}; "
                f"border: 1px solid {rel_color}; border-radius: 4px; padding: 4px 16px;"
            )
            self.reliability_label.show()
        else:
            self.reliability_label.hide()

    def _clear(self):
        """清空所有显示"""
        self.emotion_value.setText("--")
        self.emotion_confidence.setText("")
        self.stability_value.setText("--")
        self.stability_level.setText("")
        self.factor_negative_value.setText("--")
        self.factor_entropy_value.setText("--")
        self.factor_extremity_value.setText("--")
        self.quality_label.hide()
        self.reliability_label.hide()
