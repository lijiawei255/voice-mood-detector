# -*- coding: utf-8 -*-
"""
GUI 部件包 — 科研级评估报告卡片组件

本包包含用于展示结构化评估报告的可复用 PyQt5 部件，
遵循苏联构成主义设计风格（粗边框、直角、三角装饰、砖红点缀）。

组件：
- ResultCardWidget: 集成化评估结果展示（含VAD维度、稳定度分项、音频质量）
- DimensionBar: VAD 维度可视化指示条
"""

from .result_cards import ResultCardWidget, DimensionBar

__all__ = ['ResultCardWidget', 'DimensionBar']
