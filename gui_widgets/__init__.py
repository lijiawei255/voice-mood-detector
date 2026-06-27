# -*- coding: utf-8 -*-
"""
GUI 部件包 — 科研级评估报告卡片组件

本包包含用于展示结构化评估报告的可复用 PyQt5 部件，
遵循苏联构成主义设计风格（粗边框、直角、三角装饰、砖红点缀）。

组件：
- ResultCardWidget: 集成化评估结果展示（含VAD维度、稳定度分项、音频质量）
- DimensionBar: VAD 维度可视化指示条
- ResearchRadarChart: VAD 维度雷达图
- AcousticFeatureCard: 声学特征展示卡片
- PsychologicalIndicatorCard: 心理状态指标卡片
- ReliabilityBadge: 可靠性徽章
- BaselineDeviationCard: 基线偏移卡片
- ExportToolbar: 数据导出工具栏
- BaselinePanel: 个人基线管理面板
- StatsPanel: 统计分析面板
- ScoreCard: 构成主义分数卡片
- ConstructivistBackground: 构成主义几何背景层
- MplCanvas: matplotlib 趋势图表
- RecordingThread / AnalysisThread / ModelLoadThread: 后台工作线程
- ToastNotification / ToastManager: Toast 通知组件
"""

from .result_cards import ResultCardWidget, DimensionBar
from .assessment_cards import (
    AcousticFeatureCard, PsychologicalIndicatorCard,
    ReliabilityBadge, BaselineDeviationCard,
)
from .research_panel import ResearchRadarChart, ExportToolbar
from .baseline_panel import BaselinePanel
from .stats_panel import StatsPanel
from .score_card import ScoreCard
from .background import ConstructivistBackground
from .chart import MplCanvas
from .threads import RecordingThread, AnalysisThread, ModelLoadThread
from .toast import ToastNotification, ToastManager

__all__ = [
    'ResultCardWidget', 'DimensionBar',
    'AcousticFeatureCard', 'PsychologicalIndicatorCard',
    'ReliabilityBadge', 'BaselineDeviationCard',
    'ResearchRadarChart', 'ExportToolbar',
    'BaselinePanel', 'StatsPanel',
    'ScoreCard', 'ConstructivistBackground',
    'MplCanvas',
    'RecordingThread', 'AnalysisThread', 'ModelLoadThread',
    'ToastNotification', 'ToastManager',
]
