# -*- coding: utf-8 -*-
"""
语音情绪识别系统 - 图形界面模块

本模块是整个应用程序的图形用户界面（GUI）核心模块，
基于 PyQt5 实现，负责用户与用户交互、情绪检测、历史记录展示等功能。

主要功能：
1. 主窗口（MainWindow）：集成所有功能页面
2. 实时检测页面：录音控制、情绪分析、结果展示
3. 历史报告页面：历史记录列表、情绪趋势图
4. 欢迎向导对话框（WelcomeDialog）：首次运行引导
5. 数据管理对话框（DataManagerDialog）：数据清理、存储空间管理
6. 后台工作线程：录音线程、分析线程、模型加载线程
7. 自定义控件：分数卡片（ScoreCard）、趋势图（MplCanvas）、Toast通知

界面设计特点：
- 采用卡片式布局，现代化渐变设计
- 左右分栏设计，左侧为主操作区，右侧为快速指南
- 支持高 DPI 适配，中文界面清晰
- 全中文界面，用户友好

作者：Jiawei Li
许可证：GPL v3
"""

import sys
import os
import time
import logging
import shutil
from datetime import datetime
from functools import wraps

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QPushButton, QLabel, QProgressBar, QTextEdit,
    QListWidget, QSplitter, QFrame, QSizePolicy, QMessageBox, QFileDialog,
    QMenuBar, QMenu, QAction, QGridLayout, QScrollArea, QDialog, QComboBox,
    QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QObject, QUrl, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtGui import QFont, QPalette, QColor, QDesktopServices, QPainter, QBrush, QPen, QPainterPath

from recorder import AudioRecorder
from emotion_recognizer import EmotionRecognizer
from history_manager import HistoryManager
from gui_widgets.result_cards import ResultCardWidget, DimensionBar
from gui_widgets.research_panel import ResearchRadarChart, ExportToolbar
from gui_widgets.score_card import ScoreCard
from gui_widgets.background import ConstructivistBackground
from gui_widgets.chart import MplCanvas
from gui_widgets.threads import RecordingThread, AnalysisThread, ModelLoadThread
from gui_widgets.toast import ToastNotification, ToastManager
from gui_widgets.baseline_panel import BaselinePanel
from gui_widgets.stats_panel import StatsPanel
from app_paths import (
    get_recordings_dir, get_log_file, get_temp_dir,
    get_cache_dir, get_user_data_dir, get_model_cache_dir,
    get_app_dir, get_storage_stats, safe_remove_file, is_safe_path,
    load_model_config, save_model_config, is_model_downloaded
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(get_log_file(), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def exception_safe(default_return=None):
    """
    异常安全装饰器

    装饰在类方法上，自动捕获并记录异常，
    防止单个方法的异常导致整个程序崩溃。

    使用方式：
        @exception_safe(default_return=None)
        def some_method(self, ...):
            ...

    参数：
        default_return: 发生异常时的默认返回值

    返回值：
        装饰器函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"异常捕获 [{func.__name__}]: {str(e)}", exc_info=True)
                return default_return
        return wrapper
    return decorator


# 后台线程类已移至 gui_widgets.threads 模块


class WelcomeDialog(QDialog):
    """
    欢迎向导对话框类

    在首次运行程序时显示，向用户介绍：
    - 系统功能概述
    - AI 模型信息（emotion2vec+ 系列，支持 seed/base/large）
    - 技术栈说明
    - 使用方法步骤
    - 数据管理说明
    - 重要声明（非医疗诊断）

    参数：
        parent: 父窗口
        is_first_run (bool): 是否为首次运行（决定是否显示下载模型按钮）
    """
    def __init__(self, parent=None, is_first_run=True):
        super().__init__(parent)
        self.is_first_run = is_first_run
        self.selected_model = load_model_config()
        self.setWindowTitle("欢迎使用语音情绪识别系统")
        self.setMinimumSize(700, 600)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        title_label = QLabel("欢迎使用语音情绪识别系统")
        title_label.setFont(QFont("Microsoft YaHei", 22, QFont.Black))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2B2B2B; margin-bottom: 10px;")
        layout.addWidget(title_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(18)

        intro_text = QLabel(
            "<div style='line-height: 1.8; font-size: 12pt;'>"
            "<p>本系统可以通过您的语音，智能分析识别您当前的情绪状态（含混合情绪），"
            "评估情绪稳定度并给出个性化的调节建议。</p>"
            "</div>"
        )
        intro_text.setWordWrap(True)
        intro_text.setTextFormat(Qt.RichText)
        content_layout.addWidget(intro_text)

        if self.is_first_run:
            model_group = QGroupBox("选择AI模型")
            model_group.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
            model_layout = QVBoxLayout(model_group)
            model_layout.setContentsMargins(20, 25, 20, 20)
            model_text = QLabel(
                "<div style='line-height: 1.7; font-size: 11pt;'>"
                "<p><b>请选择要使用的 emotion2vec+ 模型：</b></p>"
                "<p>• 来源：ModelScope 达摩院（阿里巴巴）</p>"
                "<p>• 许可协议：Apache License 2.0（可免费使用）</p>"
                "<p style='color: #C44B4F;'><b>重要提示：</b></p>"
                "<p>首次使用需要下载模型，请确保网络连接正常。模型下载完成后，"
                "以后启动程序就会很快啦！模型会保存在程序文件夨中，不会丢失。</p>"
                "</div>"
            )
            model_text.setWordWrap(True)
            model_text.setTextFormat(Qt.RichText)
            model_layout.addWidget(model_text)
        
            # 模型选择 ComboBox
            model_select_label = QLabel("选择模型：")
            model_select_label.setFont(QFont("Microsoft YaHei", 11))
            model_layout.addWidget(model_select_label)
        
            self.model_combo = QComboBox()
            self.model_combo.setFont(QFont("Microsoft YaHei", 11))
            self.model_combo.setMinimumHeight(40)
            self.model_combo.addItem("emotion2vec_plus_large — 大型模型 (~1GB) 精度最高【推荐】", "emotion2vec_plus_large")
            self.model_combo.addItem("emotion2vec_plus_base — 基础模型 (~500MB) 速度与精度均衡", "emotion2vec_plus_base")
            self.model_combo.addItem("emotion2vec_plus_seed — 最小模型 (~200MB) 速度最快", "emotion2vec_plus_seed")
            self.model_combo.setStyleSheet("""
                QComboBox {
                    border: 2px solid #2B2B2B;
                    border-radius: 0px;
                    padding: 8px 15px;
                    background-color: #F2EDE4;
                    color: #2B2B2B;
                    font-weight: bold;
                }
                QComboBox:hover {
                    border-color: #C44B4F;
                }
                QComboBox:focus {
                    border-color: #C44B4F;
                    border-width: 3px;
                }
                QComboBox::drop-down {
                    border: none;
                    padding-right: 10px;
                }
            """)
            # 设置默认选中项
            current_model = load_model_config()
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == current_model:
                    self.model_combo.setCurrentIndex(i)
                    break
            model_layout.addWidget(self.model_combo)
            content_layout.addWidget(model_group)

        tech_group = QGroupBox("技术栈")
        tech_group.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        tech_layout = QVBoxLayout(tech_group)
        tech_layout.setContentsMargins(20, 25, 20, 20)
        tech_text = QLabel(
            "<div style='line-height: 1.7; font-size: 11pt;'>"
            "• <b>界面框架：</b>PyQt5<br>"
            "• <b>AI推理：</b>PyTorch + ModelScope<br>"
            "• <b>音频处理：</b>librosa + PyAudio<br>"
            "• <b>数据可视化：</b>matplotlib<br>"
            "• <b>便携模式：</b>所有数据保存在程序文件夹"
            "</div>"
        )
        tech_text.setWordWrap(True)
        tech_text.setTextFormat(Qt.RichText)
        tech_layout.addWidget(tech_text)
        content_layout.addWidget(tech_group)

        usage_group = QGroupBox("使用方法")
        usage_group.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        usage_layout = QVBoxLayout(usage_group)
        usage_layout.setContentsMargins(20, 25, 20, 20)
        usage_text = QLabel(
            "<div style='line-height: 2.0; font-size: 12pt;'>"
            "1. 等待模型加载完成（首次需下载，会显示进度）<br>"
            "2. 点击「开始录音」按钮<br>"
            "3. 自然地说出您现在的感受（建议 3-30 秒）<br>"
            "4. 点击「停止录音」按钮<br>"
            "5. 稍等几秒，即可查看分析结果<br><br>"
            "<b>提示：</b>在安静的环境下录音，效果更佳。"
            "</div>"
        )
        usage_text.setWordWrap(True)
        usage_text.setTextFormat(Qt.RichText)
        usage_layout.addWidget(usage_text)
        content_layout.addWidget(usage_group)

        tips_group = QGroupBox("数据管理")
        tips_group.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        tips_layout = QVBoxLayout(tips_group)
        tips_layout.setContentsMargins(20, 25, 20, 20)
        tips_text = QLabel(
            "<div style='line-height: 1.8; font-size: 11pt;'>"
            "<p><b>您的数据都在这里：</b></p>"
            "• 点击顶部菜单「文件 → 数据管理」，可以查看、删除历史记录<br>"
            "• 可以清除临时文件、日志、缓存等无用文件<br>"
            "• 您的录音和历史记录可以随时删除<br>"
            "• <b>AI模型文件请不要删除</b>（约1GB，删除后需要重新下载）"
            "</div>"
        )
        tips_text.setWordWrap(True)
        tips_text.setTextFormat(Qt.RichText)
        tips_layout.addWidget(tips_text)
        content_layout.addWidget(tips_group)

        disclaimer_group = QGroupBox("重要声明")
        disclaimer_group.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        disclaimer_layout = QVBoxLayout(disclaimer_group)
        disclaimer_layout.setContentsMargins(20, 25, 20, 20)
        disclaimer_text = QLabel(
            "<div style='line-height: 1.8; font-size: 11pt; color: #c0392b;'>"
            "<p><b>本软件仅供个人非商用参考使用！</b></p>"
            "<p>检测结果仅作为情绪状态的辅助参考，<b>不构成任何医疗诊断或建议</b>。</p>"
            "<p>如果您感到持续的情绪低落、焦虑或其他心理困扰，"
            "请及时咨询专业的心理医生或相关专业人士。</p>"
            "</div>"
        )
        disclaimer_text.setWordWrap(True)
        disclaimer_text.setTextFormat(Qt.RichText)
        disclaimer_layout.addWidget(disclaimer_text)
        content_layout.addWidget(disclaimer_group)

        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        if self.is_first_run:
            start_btn = QPushButton("开始使用（下载模型）")
            start_btn.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
            start_btn.setMinimumHeight(50)
            start_btn.setMinimumWidth(250)
            start_btn.setStyleSheet("""
                QPushButton {
                    background-color: #C44B4F;
                    color: white;
                    border: 2px solid #2B2B2B;
                    border-radius: 0px;
                    padding: 12px 30px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #A33B3F;
                }
                QPushButton:pressed {
                    background-color: #8A2E31;
                }
            """)
            start_btn.clicked.connect(self._on_start_clicked)
            btn_layout.addWidget(start_btn)
        else:
            close_btn = QPushButton("知道了")
            close_btn.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
            close_btn.setMinimumHeight(45)
            close_btn.setMinimumWidth(150)
            close_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2B2B2B;
                    color: white;
                    border: 2px solid #2B2B2B;
                    border-radius: 0px;
                    padding: 10px 30px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #C44B4F;
                    border-color: #C44B4F;
                }
            """)
            close_btn.clicked.connect(self.accept)
            btn_layout.addWidget(close_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _on_start_clicked(self):
        """首次运行时点击“开始使用”按钮的处理"""
        if hasattr(self, 'model_combo'):
            self.selected_model = self.model_combo.currentData()
            save_model_config(self.selected_model)
        self.accept()

    @staticmethod
    def is_model_downloaded():
        """ 检查当前配置的模型是否已下载 """
        current_model = load_model_config()
        return is_model_downloaded(current_model)


class ModelSwitchDialog(QDialog):
    """
    模型切换对话框类

    允许用户在运行时切换不同的 emotion2vec+ 模型。
    显示每个模型的信息、下载状态和当前使用状态。
    """
    def __init__(self, current_model_name, parent=None):
        super().__init__(parent)
        self.current_model = current_model_name
        self.selected_model = current_model_name
        self.setWindowTitle("切换AI模型")
        self.setMinimumSize(550, 420)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(25, 25, 25, 25)

        title_label = QLabel("选择情绪识别模型")
        title_label.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2B2B2B; margin-bottom: 5px;")
        layout.addWidget(title_label)

        desc_label = QLabel("选择不同的模型会影响识别精度和运行速度，切换后需要重新加载模型。")
        desc_label.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        desc_label.setStyleSheet("color: #8A8580;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        from emotion_recognizer import AVAILABLE_MODELS

        self.btn_group = QButtonGroup(self)
        models_info = [
            ("emotion2vec_plus_large", "Large（大型模型）", "~1GB", "精度最高，推荐使用"),
            ("emotion2vec_plus_base", "Base（基础模型）", "~500MB", "速度与精度均衡"),
            ("emotion2vec_plus_seed", "Seed（最小模型）", "~200MB", "速度最快，适合低配置设备"),
        ]

        for model_name, display, size, desc in models_info:
            downloaded = is_model_downloaded(model_name)
            is_current = (model_name == self.current_model)

            radio = QRadioButton()
            radio.setProperty("model_name", model_name)
            if is_current:
                radio.setChecked(True)

            # 构建卡片展示文本
            status_text = "■ 当前使用" if is_current else ("■ 已下载" if downloaded else "△ 需要下载")
            status_color = "#2B2B2B" if (is_current or downloaded) else "#C44B4F"
            radio.setText(f"{display}  |  大小: {size}  |  {desc}")
            radio.setFont(QFont("Microsoft YaHei", 11))
            radio.setMinimumHeight(50)
            radio.setStyleSheet(f"""
                QRadioButton {{
                    padding: 12px 15px;
                    border: 2px solid {'#C44B4F' if is_current else '#2B2B2B'};
                    border-radius: 0px;
                    background-color: {'#F2EDE4' if is_current else '#FFFFFF'};
                    font-weight: bold;
                }}
                QRadioButton:hover {{
                    border-color: #C44B4F;
                    background-color: #F2EDE4;
                }}
            """)

            self.btn_group.addButton(radio)
            layout.addWidget(radio)

            # 状态标签
            status_label = QLabel(f"    {status_text}")
            status_label.setFont(QFont("Microsoft YaHei", 9))
            status_label.setStyleSheet(f"color: {status_color}; margin-left: 30px; margin-top: -5px;")
            layout.addWidget(status_label)

        layout.addStretch()

        # 按钮区域
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.setFont(QFont("Microsoft YaHei", 11))
        cancel_btn.setMinimumHeight(40)
        cancel_btn.setMinimumWidth(100)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #E8E3DA;
                color: #2B2B2B;
                border: 2px solid #2B2B2B;
                border-radius: 0px;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #8A8580;
                color: white;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self.confirm_btn = QPushButton("确认切换")
        self.confirm_btn.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.confirm_btn.setMinimumHeight(40)
        self.confirm_btn.setMinimumWidth(140)
        self.confirm_btn.setStyleSheet("""
            QPushButton {
                background-color: #C44B4F;
                color: white;
                border: 2px solid #2B2B2B;
                border-radius: 0px;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #A33B3F;
            }
        """)
        self.confirm_btn.clicked.connect(self._on_confirm)
        btn_layout.addWidget(self.confirm_btn)

        layout.addLayout(btn_layout)

    def _on_confirm(self):
        """确认切换按钮处理"""
        checked = self.btn_group.checkedButton()
        if checked:
            self.selected_model = checked.property("model_name")
        self.accept()

    def get_selected_model(self):
        """获取用户选择的模型名"""
        return self.selected_model


class DataManagerDialog(QDialog):
    """
    数据管理对话框类

    提供数据管理功能，包括：
    - 显示数据存储位置和目录结构说明
    - 存储空间统计（录音、模型、日志、临时文件、缓存等）
    - 清空历史记录和录音文件
    - 一键清理系统垃圾（临时文件、缓存、日志）
    - 打开数据文件夹

    参数：
        parent: 父窗口（主窗口）
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("数据管理")
        self.setMinimumSize(700, 550)
        self.resize(750, 650)
        self.setup_ui()
        self.refresh_stats()

    def setup_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setSpacing(10)
        outer_layout.setContentsMargins(15, 15, 15, 15)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setSpacing(15)
        layout.setContentsMargins(5, 5, 5, 5)

        info_group = QGroupBox("数据存储位置（便携模式）")
        info_group.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        info_layout = QVBoxLayout(info_group)

        self.path_info = QLabel("")
        self.path_info.setTextFormat(Qt.RichText)
        self.path_info.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.path_info.setOpenExternalLinks(True)
        self.path_info.setWordWrap(True)
        info_layout.addWidget(self.path_info)

        layout.addWidget(info_group)

        self.refresh_path_info()

        self.stats_label = QLabel("")
        self.stats_label.setFont(QFont("Microsoft YaHei", 11))
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("padding: 12px; background-color: #F2EDE4; border-radius: 0px; border: 2px solid #2B2B2B; font-weight: bold;")
        layout.addWidget(self.stats_label)

        actions_group = QGroupBox("数据清理操作")
        actions_group.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        actions_layout = QVBoxLayout(actions_group)
        actions_layout.setSpacing(10)
        actions_layout.setContentsMargins(15, 25, 15, 15)

        del_all_btn = QPushButton("清空所有历史记录和录音")
        del_all_btn.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        del_all_btn.setMinimumHeight(45)
        del_all_btn.setObjectName("dangerBtn")
        del_all_btn.setCursor(Qt.PointingHandCursor)
        del_all_btn.clicked.connect(self.delete_all_records)
        actions_layout.addWidget(del_all_btn)

        clear_all_btn = QPushButton("一键清理系统垃圾")
        clear_all_btn.setFont(QFont("Microsoft YaHei", 11))
        clear_all_btn.setMinimumHeight(45)
        clear_all_btn.setCursor(Qt.PointingHandCursor)
        clear_all_btn.setToolTip("一键清除临时文件、缓存文件和日志文件")
        clear_all_btn.clicked.connect(self.clear_all_junk)
        actions_layout.addWidget(clear_all_btn)

        open_folder_btn = QPushButton("打开数据文件夹")
        open_folder_btn.setFont(QFont("Microsoft YaHei", 11))
        open_folder_btn.setMinimumHeight(45)
        open_folder_btn.setCursor(Qt.PointingHandCursor)
        open_folder_btn.clicked.connect(self.open_data_folder)
        actions_layout.addWidget(open_folder_btn)

        layout.addWidget(actions_group)
        layout.addStretch()

        scroll_area.setWidget(scroll_content)
        outer_layout.addWidget(scroll_area, 1)

        self.dm_refresh_btn = QPushButton("刷新统计")
        self.dm_refresh_btn.setFont(QFont("Microsoft YaHei", 10))
        self.dm_refresh_btn.setMinimumHeight(40)
        self.dm_refresh_btn.setMinimumWidth(120)
        self.dm_refresh_btn.setCursor(Qt.PointingHandCursor)
        self.dm_refresh_btn.clicked.connect(self._on_dm_refresh_clicked)

        close_btn = QPushButton("关闭")
        close_btn.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        close_btn.setMinimumHeight(40)
        close_btn.setMinimumWidth(120)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.clicked.connect(self.accept)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.dm_refresh_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        outer_layout.addLayout(btn_layout)

    def _on_dm_refresh_clicked(self):
        """数据管理刷新按钮点击（带动画效果）"""
        try:
            self.dm_refresh_btn.setText("刷新中...")
            self.dm_refresh_btn.setEnabled(False)
            # 统计区域闪烁效果
            original_style = self.stats_label.styleSheet()
            self.stats_label.setStyleSheet("padding: 12px; background-color: #E8E3DA; border-radius: 0px; border: 2px solid #C44B4F; font-weight: bold;")
            from PyQt5.QtCore import QTimer
            def _restore():
                self.stats_label.setStyleSheet(original_style)
                self.refresh_stats()
                self.dm_refresh_btn.setText("刷新统计")
                self.dm_refresh_btn.setEnabled(True)
            QTimer.singleShot(300, _restore)
        except Exception as e:
            logger.error(f"刷新动画异常: {str(e)}")
            self.refresh_stats()
            self.dm_refresh_btn.setText("刷新统计")
            self.dm_refresh_btn.setEnabled(True)

    def refresh_path_info(self):
        """刷新数据存储路径显示（动态适配不同根目录）"""
        try:
            app_dir = get_app_dir()
            data_dir = get_user_data_dir()

            def path_to_url(p):
                return QUrl.fromLocalFile(os.path.abspath(p)).toString()

            path_html = (
                f"<div style='line-height: 1.8; font-size: 11pt;'>"
                f"<p><b>程序目录：</b><br>"
                f"<a href='{path_to_url(app_dir)}' style='color: #C44B4F;'>{app_dir}</a></p>"
                f"<p><b>数据目录（所有用户数据保存在此）：</b><br>"
                f"<a href='{path_to_url(data_dir)}' style='color: #C44B4F;'>{data_dir}</a></p>"
                f"<p><b>数据结构说明：</b></p>"
                f"<ul style='margin: 5px 0; padding-left: 20px; line-height: 2.0;'>"
                f"<li><b>recordings/</b> —— 您的录音文件（可删除）</li>"
                f"<li><b>models/</b> —— AI模型文件（<span style='color: #C44B4F; font-weight: bold;'>不可删除，删除后需重新下载约1GB</span>）</li>"
                f"<li><b>logs/</b> —— 程序运行日志（可清除）</li>"
                f"<li><b>temp/</b> —— 临时文件（可清除）</li>"
                f"<li><b>cache/</b> —— 缓存文件（可清除）</li>"
                f"<li><b>history.json</b> —— 历史记录（可删除）</li>"
                f"</ul>"
                f"</div>"
            )
            self.path_info.setText(path_html)
        except Exception as e:
            logger.error(f"刷新路径信息失败: {str(e)}")

    def refresh_stats(self):
        try:
            self.refresh_path_info()
            stats = get_storage_stats()
            stats_text = "<b>存储统计：</b><br>"
            for key, name in [('recordings', '录音文件'), ('models', 'AI模型'),
                              ('logs', '日志'), ('temp', '临时文件'), ('cache', '缓存')]:
                if key in stats:
                    s = stats[key]
                    stats_text += f"• {name}: {s['file_count']} 个文件, {s['size_mb']:.2f} MB<br>"
            if 'history' in stats:
                stats_text += f"• 历史记录文件: {stats['history']['size_mb']:.2f} MB"
            self.stats_label.setText(stats_text)
            self.stats_label.setTextFormat(Qt.RichText)
        except Exception as e:
            logger.error(f"刷新统计失败: {str(e)}")

    def delete_all_records(self):
        reply = QMessageBox.question(
            self, "确认删除",
            "确定要清空所有历史记录和录音文件吗？\n\n"
            "此操作不可恢复！\n"
            "（AI模型文件不会被删除）",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            if self.parent_window:
                self.parent_window.clear_all_history()
                self.refresh_stats()

    def _clear_directory(self, dir_path, name):
        count = 0
        size_freed = 0
        try:
            if not os.path.exists(dir_path):
                return 0, 0
            for filename in os.listdir(dir_path):
                filepath = os.path.join(dir_path, filename)
                try:
                    if os.path.isfile(filepath):
                        size = os.path.getsize(filepath)
                        if safe_remove_file(filepath):
                            count += 1
                            size_freed += size
                    elif os.path.isdir(filepath):
                        if is_safe_path(filepath):
                            folder_size = 0
                            for root, dirs, files in os.walk(filepath):
                                for f in files:
                                    try:
                                        fp = os.path.join(root, f)
                                        folder_size += os.path.getsize(fp)
                                    except OSError:
                                        pass
                            shutil.rmtree(filepath, ignore_errors=True)
                            size_freed += folder_size
                except Exception as e:
                    logger.warning(f"删除文件失败 {filepath}: {str(e)}")
            return count, size_freed
        except Exception as e:
            logger.error(f"清理{name}失败: {str(e)}")
            return count, size_freed

    def clear_all_junk(self):
        try:
            total_count = 0
            total_size = 0

            temp_dir = get_temp_dir()
            c1, s1 = self._clear_directory(temp_dir, "临时文件")
            total_count += c1
            total_size += s1

            cache_dir = get_cache_dir()
            c2, s2 = self._clear_directory(cache_dir, "缓存文件")
            total_count += c2
            total_size += s2

            log_file = get_log_file()
            log_cleared = False
            if os.path.exists(log_file):
                try:
                    log_size = os.path.getsize(log_file)
                    with open(log_file, 'w', encoding='utf-8') as f:
                        f.write('')
                    total_size += log_size
                    log_cleared = True
                    total_count += 1
                except OSError:
                    pass
            logs_dir = os.path.dirname(log_file)
            if logs_dir and os.path.exists(logs_dir) and logs_dir != temp_dir and logs_dir != cache_dir:
                c3, s3 = self._clear_directory(logs_dir, "日志文件")
                total_count += c3
                total_size += s3

            size_mb = total_size / (1024 * 1024)
            msg = f"系统垃圾清理完成！\n\n"
            msg += f"清理文件数：{total_count} 个\n"
            msg += f"释放空间：{size_mb:.2f} MB"
            if log_cleared:
                msg += "\n\n日志文件已重置"
            QMessageBox.information(self, "清理完成", msg)

            if self.parent_window:
                self.parent_window.append_log(f"一键清理完成，清理 {total_count} 个文件，释放 {size_mb:.2f} MB")
            self.refresh_stats()
        except Exception as e:
            logger.error(f"一键清理失败: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "错误", f"清理失败: {str(e)}")

    def open_data_folder(self):
        data_dir = get_user_data_dir()
        try:
            os.makedirs(data_dir, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(data_dir))
        except Exception as e:
            QMessageBox.warning(self, "提示", f"无法打开文件夹: {str(e)}\n\n路径: {data_dir}")



class DiagonalStripe(QWidget):
    """构成主义对角线覆盖层 — 单条粗斜线贯穿界面"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()
        # 单条贯穿对角线：砖红色，4px实线
        pen = QPen(QColor(196, 75, 79, 90))
        pen.setWidth(4)
        pen.setCapStyle(Qt.FlatCap)
        p.setPen(pen)
        p.drawLine(0, 0, w, h)
        p.end()


class MainWindow(QMainWindow):
    """
    主窗口类

    应用程序的主界面窗口，集成所有功能模块：
    - 菜单栏：文件菜单（数据管理、退出）、帮助菜单
    - 实时检测页面：录音控制、情绪分析、结果展示
    - 历史报告页面：历史记录列表、情绪趋势图
    - 底部操作区：数据管理按钮、版权声明
    - Toast通知：错误/警告时弹窗提示

    工作流程：
    1. 启动时显示欢迎向导（首次运行）
    2. 后台加载 AI 模型
    3. 模型加载完成后启用录音功能
    4. 用户录音 → 停止 → 分析 → 显示结果 → 保存历史
    5. 可随时查看历史记录和趋势图

    信号：
    - log_signal: 日志输出信号（线程安全）
    - model_loaded_signal: 模型加载完成信号
    - model_progress_signal: 模型加载进度信号
    """
    log_signal = pyqtSignal(str)
    model_loaded_signal = pyqtSignal(bool, str)
    model_progress_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.recorder = AudioRecorder()
        self.recording_thread = None
        self.analysis_thread = None
        self.model_load_thread = None
        self.is_recording = False
        self.is_analyzing = False
        self.current_audio_path = None
        self.record_start_time = None
        self.recognizer = None
        self.history_manager = HistoryManager(cleanup_callback=self._on_history_cleanup_needed)
        self.temp_files = []
        self._closing = False

        self.log_signal.connect(self._append_log_safe)
        self.model_loaded_signal.connect(self._on_model_loaded_safe)
        self.model_progress_signal.connect(self._append_log_safe)

        self.toast_manager = ToastManager(self)

        self.init_ui()
        self.setup_menubar()
        self.setup_styles()
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        QTimer.singleShot(100, self.check_first_run)
        QTimer.singleShot(200, self.load_model_async)

    @exception_safe()
    def check_first_run(self):
        if not WelcomeDialog.is_model_downloaded():
            self.append_log("检测到首次运行，正在显示欢迎向导...")
            dialog = WelcomeDialog(self, is_first_run=True)
            dialog.exec_()

    def _append_log_safe(self, msg):
        """ 记录日志，仅写入logger，不再显示在UI控件中 """
        try:
            logger.info(msg)
        except Exception:
            pass

    def append_log(self, msg):
        """ 记录日志并在错误/警告时显示Toast通知 """
        logger.info(msg)
        # 仅对错误和警告显示Toast弹窗
        msg_lower = msg.lower()
        if '错误' in msg or '失败' in msg or 'error' in msg_lower or '异常' in msg:
            if hasattr(self, 'toast_manager'):
                self.toast_manager.show_toast(msg, level="error", duration=6000)
        elif '警告' in msg or 'warning' in msg_lower or '注意' in msg:
            if hasattr(self, 'toast_manager'):
                self.toast_manager.show_toast(msg, level="warning", duration=4000)

    def _on_model_loaded_safe(self, success, error):
        if self._closing:
            return
        # 重新启用模型切换按钮
        self.model_switch_btn.setEnabled(True)
        if success:
            self._append_log_safe("情绪识别模型加载完成，准备就绪！")
            self.status_label.setText("● 准备就绪")
            self.status_label.setStyleSheet("color: #FFFFFF; background-color: #2B2B2B; padding: 8px 18px; border-radius: 0px; border: 2px solid #2B2B2B; font-weight: bold;")
            self.record_btn.setText("● 点击开始录音")
            self.record_btn.setProperty("isRecording", "false")
            self.record_btn.style().unpolish(self.record_btn)
            self.record_btn.style().polish(self.record_btn)
            self.record_btn.setEnabled(True)
            logger.info("情绪识别模型加载成功")
        else:
            error_msg = f"模型加载失败: {error}"
            self._append_log_safe(f"{error_msg}")
            logger.error(error_msg)
            self.status_label.setText("● 模型加载失败")
            self.status_label.setStyleSheet("color: #FFFFFF; background-color: #C44B4F; padding: 8px 18px; border-radius: 0px; border: 2px solid #2B2B2B; font-weight: bold;")
            QMessageBox.warning(
                self, "模型加载提示",
                f"情绪识别模型加载失败:\n{error}\n\n"
                "请检查：\n"
                "1. 网络连接是否正常（首次需要下载模型）\n"
                "2. 磁盘空间是否充足\n"
                "3. 程序文件夹是否有写入权限\n\n"
                "您可以尝试重启程序或切换其他模型"
            )

    @exception_safe()
    def load_model_async(self):
        self.append_log("系统启动中，正在准备界面...")
        self.append_log(f"程序目录: {get_app_dir()}")
        self.append_log(f"数据目录: {get_user_data_dir()}")

        current_model = load_model_config()
        from emotion_recognizer import AVAILABLE_MODELS
        model_display = AVAILABLE_MODELS.get(current_model, {}).get('display', current_model)
        self.append_log(f"当前选择模型: {model_display}")
        self.append_log("正在后台加载情绪识别模型，请稍候...")
        self.append_log("（首次使用需要下载模型，下载完成后下次启动会很快）")

        try:
            def on_progress(msg):
                if not self._closing:
                    self.model_progress_signal.emit(msg)

            self.recognizer = EmotionRecognizer(progress_callback=on_progress, model_name=current_model)

            def on_loaded(success, error):
                if not self._closing:
                    self.model_loaded_signal.emit(success, error or "")

            self.recognizer.load_model(callback=on_loaded)
        except Exception as e:
            logger.error(f"模型初始化错误: {str(e)}", exc_info=True)
            self.append_log(f"模型初始化错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"程序初始化失败:\n{str(e)}")

    @exception_safe()
    def init_ui(self):
        self.setWindowTitle("语音情绪识别系统 v1.1 - 便携版")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 构成主义几何背景
        self._bg_widget = ConstructivistBackground(central_widget)
        self._bg_widget.setGeometry(central_widget.rect())
        self._bg_widget.lower()

        # 构成主义对角线覆盖层 — 前景绘制，贯穿所有控件
        self._diag_stripe = DiagonalStripe(central_widget)
        self._diag_stripe.setGeometry(central_widget.rect())
        self._diag_stripe.raise_()

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")
        header_frame.setMinimumHeight(70)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(25, 12, 25, 12)

        # 标题左侧大型红色楔形块（构成主义标志性元素）
        title_accent = QLabel()
        title_accent.setFixedSize(18, 34)
        title_accent.setStyleSheet("background-color: #C44B4F; border: 2px solid #2B2B2B;")
        header_layout.addWidget(title_accent)
        # 红色楔形与标题之间的细黑分隔条
        sep = QLabel()
        sep.setFixedSize(8, 34)
        sep.setStyleSheet("background-color: #2B2B2B;")
        header_layout.addWidget(sep)

        title_label = QLabel("■ 语音情绪识别系统")
        title_label.setFont(QFont("Microsoft YaHei", 30, QFont.Black))
        title_label.setObjectName("headerTitle")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        # 模型切换按钮
        self.model_switch_btn = QPushButton("切换模型")
        self.model_switch_btn.setFont(QFont("Microsoft YaHei", 10))
        self.model_switch_btn.setCursor(Qt.PointingHandCursor)
        self.model_switch_btn.setStyleSheet("""
            QPushButton {
                background-color: #F2EDE4;
                color: #2B2B2B;
                border: 2px solid #2B2B2B;
                border-radius: 0px;
                padding: 6px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2B2B2B;
                color: #FFFFFF;
                border-color: #2B2B2B;
            }
        """)
        self.model_switch_btn.clicked.connect(self.show_model_switch_dialog)
        header_layout.addWidget(self.model_switch_btn)

        self.status_label = QLabel("● 正在启动...")
        self.status_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.status_label.setObjectName("statusLoading")
        self.status_label.setStyleSheet("color: #FFFFFF; background-color: #8A8580; padding: 8px 18px; border-radius: 0px; border: 2px solid #2B2B2B; font-weight: bold;")
        header_layout.addWidget(self.status_label)

        main_layout.addWidget(header_frame)

        self.tab_widget = QTabWidget()
        self.tab_widget.setFont(QFont("Microsoft YaHei", 11))
        self.tab_widget.setObjectName("mainTab")

        self.create_realtime_tab()
        self.create_history_tab()

        main_layout.addWidget(self.tab_widget, 1)

        self.create_bottom_buttons(main_layout)

    @exception_safe()
    def setup_menubar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件")

        data_mgr_action = QAction("数据管理", self)
        data_mgr_action.setShortcut("Ctrl+D")
        data_mgr_action.triggered.connect(self.show_data_manager)
        file_menu.addAction(data_mgr_action)

        file_menu.addSeparator()

        exit_action = QAction("退出", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    @exception_safe()
    def show_data_manager(self, checked=False):
        dialog = DataManagerDialog(self)
        dialog.exec_()

    @exception_safe()
    def show_model_switch_dialog(self, checked=False):
        """显示模型切换对话框，允许用户在运行时切换模型"""
        if not self.recognizer:
            QMessageBox.warning(self, "提示", "识别器未初始化，请稍候")
            return
        if self.is_recording or self.is_analyzing:
            QMessageBox.warning(self, "提示", "请等当前录音/分析完成后再切换模型")
            return
        if self.recognizer.loading:
            QMessageBox.warning(self, "提示", "模型正在加载中，请稍候")
            return

        current_model = self.recognizer.model_name
        dialog = ModelSwitchDialog(current_model, self)
        if dialog.exec_() == QDialog.Accepted:
            target_model = dialog.get_selected_model()
            if target_model == current_model and self.recognizer.loaded:
                self.append_log("当前已是该模型，无需切换")
                return
            self._do_switch_model(target_model)

    def _do_switch_model(self, target_model):
        """执行模型切换逻辑"""
        from emotion_recognizer import AVAILABLE_MODELS
        model_info = AVAILABLE_MODELS.get(target_model, {})
        model_display = model_info.get('display', target_model)
        model_size = model_info.get('size', '未知')

        # 检查目标模型是否已下载
        if not is_model_downloaded(target_model):
            reply = QMessageBox.question(
                self, "模型下载确认",
                f"模型 [{model_display}] 尚未下载。\n\n"
                f"需要下载约 {model_size} 的模型文件，是否继续？\n"
                f"（请确保网络连接正常）",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                self.append_log("用户取消模型切换")
                return

        # 开始切换
        self.append_log(f"正在切换到模型 [{model_display}]...")
        self.record_btn.setEnabled(False)
        self.model_switch_btn.setEnabled(False)
        self.status_label.setText("● 正在切换模型...")
        self.status_label.setStyleSheet("color: #FFFFFF; background-color: #8A8580; padding: 8px 18px; border-radius: 0px; border: 2px solid #2B2B2B; font-weight: bold;")

        def on_progress(msg):
            if not self._closing:
                self.model_progress_signal.emit(msg)

        # 确保进度回调生效
        self.recognizer._progress_callbacks = [on_progress]

        def on_switch_done(success, error):
            if self._closing:
                return
            self.model_loaded_signal.emit(success, error or "")
            if success:
                # 更新底部版权标签
                try:
                    self.credit_label.setText(
                        f"Powered by {target_model} (ModelScope/达摩院, Apache-2.0) | "
                        "便携模式 - 数据保存在程序目录 | 仅供个人非商用参考使用"
                    )
                except Exception:
                    pass

        self.recognizer.switch_model(target_model, callback=on_switch_done)

    @exception_safe()
    def create_realtime_tab(self):
        realtime_widget = QWidget()
        layout = QVBoxLayout(realtime_widget)
        layout.setSpacing(12)

        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setObjectName("mainSplitter")

        left_panel = QFrame()
        left_panel.setObjectName("leftPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(12)
        left_layout.setContentsMargins(5, 5, 5, 5)

        control_group = QGroupBox("■ 录音控制")
        control_group.setFont(QFont("Microsoft YaHei", 12, QFont.Black))
        control_group.setObjectName("controlGroup")
        control_group.setStyleSheet("""
            QGroupBox#controlGroup {
                border: 3px solid #2B2B2B;
                border-top: 6px solid #C44B4F;
                border-radius: 0px;
                margin-top: 14px;
                padding-top: 22px;
                background-color: #F2EDE4;
            }
            QGroupBox#controlGroup::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 2px 14px;
                color: #C44B4F;
                font-weight: 900;
            }
        """)
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(12)
        control_layout.setContentsMargins(20, 25, 20, 20)

        self.record_btn = QPushButton("● 模型加载中...")
        self.record_btn.setFont(QFont("Microsoft YaHei", 22, QFont.Black))
        self.record_btn.setMinimumHeight(130)
        self.record_btn.setObjectName("recordBtn")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        self.record_btn.setCursor(Qt.PointingHandCursor)
        control_layout.addWidget(self.record_btn)

        info_grid = QGridLayout()
        info_grid.setSpacing(10)
        info_grid.setColumnStretch(0, 2)
        info_grid.setColumnStretch(1, 3)

        duration_title = QLabel("■ 录音时长")
        duration_title.setFont(QFont("Microsoft YaHei", 11, QFont.Black))
        info_grid.addWidget(duration_title, 0, 0)

        self.duration_display = QLabel("00:00")
        self.duration_display.setFont(QFont("Consolas", 28, QFont.Black))
        self.duration_display.setMinimumWidth(120)
        self.duration_display.setAlignment(Qt.AlignCenter)
        self.duration_display.setObjectName("durationLabel")
        info_grid.addWidget(self.duration_display, 0, 1)

        # P0 新增：录音质量实时反馈
        quality_title = QLabel("■ 录音质量")
        quality_title.setFont(QFont("Microsoft YaHei", 11, QFont.Black))
        info_grid.addWidget(quality_title, 1, 0)
        self.quality_feedback = QLabel("等待录音...")
        self.quality_feedback.setFont(QFont("Microsoft YaHei", 10))
        self.quality_feedback.setAlignment(Qt.AlignCenter)
        self.quality_feedback.setMinimumHeight(28)
        self.quality_feedback.setStyleSheet(
            "background-color: #F2EDE4; border: 2px solid #2B2B2B; "
            "padding: 4px; color: #2B2B2B; font-weight: bold;"
        )
        info_grid.addWidget(self.quality_feedback, 1, 1)

        progress_title = QLabel("■ 处理进度")
        progress_title.setFont(QFont("Microsoft YaHei", 11, QFont.Black))
        info_grid.addWidget(progress_title, 2, 0)

        self.record_progress = QProgressBar()
        self.record_progress.setRange(0, 100)
        self.record_progress.setValue(0)
        self.record_progress.setObjectName("recordProgress")
        self.record_progress.setTextVisible(False)
        self.record_progress.setMinimumHeight(24)
        info_grid.addWidget(self.record_progress, 2, 1)

        control_layout.addLayout(info_grid)
        left_layout.addWidget(control_group)

        result_group = QGroupBox("■ 检测结果")
        result_group.setFont(QFont("Microsoft YaHei", 12, QFont.Black))
        result_group.setObjectName("resultGroup")
        result_group.setStyleSheet("""
            QGroupBox#resultGroup {
                border: 3px solid #2B2B2B;
                border-top: 6px solid #C44B4F;
                border-radius: 0px;
                margin-top: 14px;
                padding-top: 22px;
                background-color: #F2EDE4;
            }
            QGroupBox#resultGroup::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 2px 14px;
                color: #C44B4F;
                font-weight: 900;
            }
        """)
        self.result_group = result_group
        result_layout = QVBoxLayout(result_group)
        result_layout.setSpacing(18)
        result_layout.setContentsMargins(25, 25, 25, 25)

        # 构成主义不对称卡片布局：左侧大卡片(2列宽) + 右侧3小卡片堆叠
        cards_grid = QGridLayout()
        cards_grid.setSpacing(12)

        self.score_card = ScoreCard("情绪稳定度 (0-10)", accent_color="#C44B4F")
        self.score_card.setMinimumHeight(220)
        self.level_card = ScoreCard("情绪状态", accent_color="#C44B4F")
        self.emotion_card = ScoreCard("主要情绪", accent_color="#C44B4F")
        self.compound_card = ScoreCard("复合情绪", accent_color="#C44B4F")

        # 网格布局：左侧大卡片占2行2列，右侧3张小卡片各占1行1列
        cards_grid.addWidget(self.score_card, 0, 0, 3, 2)    # 占3行2列
        cards_grid.addWidget(self.level_card, 0, 2, 1, 1)    # 第1行第3列
        cards_grid.addWidget(self.emotion_card, 1, 2, 1, 1)  # 第2行第3列
        cards_grid.addWidget(self.compound_card, 2, 2, 1, 1) # 第3行第3列

        result_layout.addLayout(cards_grid)

        # P0 新增：集成化科研评估报告卡片（VAD维度 + 稳定度分项 + 音频质量）
        self.result_card_widget = ResultCardWidget()
        self.result_card_widget.hide()
        result_layout.addWidget(self.result_card_widget)

        # P1 新增：VAD 维度雷达图
        self.radar_chart = ResearchRadarChart()
        self.radar_chart.hide()
        result_layout.addWidget(self.radar_chart)

        # 分数说明
        stability_hint = QLabel("■ 分数越低越稳定 · 0分最稳定 · 10分波动最大")
        stability_hint.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        stability_hint.setAlignment(Qt.AlignCenter)
        stability_hint.setStyleSheet("color: #2B2B2B; padding: 6px 0; font-weight: bold; background: #E8E3DA; border: 1px solid #2B2B2B;")
        result_layout.addWidget(stability_hint)

        self.warning_label = QLabel("")
        self.warning_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.warning_label.setAlignment(Qt.AlignCenter)
        self.warning_label.setObjectName("warningLabel")
        self.warning_label.setMinimumHeight(50)
        self.warning_label.setWordWrap(True)
        self.warning_label.hide()
        result_layout.addWidget(self.warning_label)

        prob_title = QLabel("■ 情绪概率分布")
        prob_title.setFont(QFont("Microsoft YaHei", 13, QFont.Black))
        prob_title.setStyleSheet("color: #C44B4F; padding: 6px 0; font-weight: 900;")
        result_layout.addWidget(prob_title)

        self.prob_bars = {}
        self.prob_labels = {}
        emotions = [("平静", "#C44B4F"), ("开心", "#C44B4F"), ("惊讶", "#C44B4F"),
                    ("悲伤", "#C44B4F"), ("愤怒", "#C44B4F"), ("恐惧", "#C44B4F"),
                    ("厌恶", "#C44B4F")]
        for idx, (emotion, color) in enumerate(emotions):
            bar_row = QHBoxLayout()
            bar_row.setSpacing(12)

            label = QLabel(emotion)
            label.setFont(QFont("Microsoft YaHei", 11, QFont.Black))
            label.setMinimumWidth(55)
            label.setMaximumWidth(55)
            label.setStyleSheet("color: #2B2B2B; font-weight: 900;")
            bar_row.addWidget(label)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setObjectName(f"probBar_{emotion}")
            bar.setMinimumHeight(32)
            bar.setMaximumHeight(32)
            bar.setProperty("barColor", color)
            self.prob_bars[emotion] = bar
            bar_row.addWidget(bar, 1)

            pct_label = QLabel("0%")
            pct_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
            pct_label.setMinimumWidth(55)
            pct_label.setMaximumWidth(55)
            pct_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            pct_label.setObjectName(f"probLabel_{emotion}")
            self.prob_labels[emotion] = pct_label
            bar_row.addWidget(pct_label)

            result_layout.addLayout(bar_row)


        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setWidget(result_group)
        left_layout.addWidget(scroll_area, 1)

        content_splitter.addWidget(left_panel)

        right_panel = QFrame()
        right_panel.setObjectName("rightPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(12)
        right_layout.setContentsMargins(5, 5, 5, 5)

        quick_guide = QGroupBox("▸ 快速指南")
        quick_guide.setFont(QFont("Microsoft YaHei", 12, QFont.Black))
        quick_guide.setObjectName("guideGroup")
        quick_guide.setStyleSheet("""
            QGroupBox#guideGroup {
                border: 3px solid #2B2B2B;
                border-radius: 0px;
                margin-top: 14px;
                padding-top: 22px;
                background-color: #F2EDE4;
                font-family: 'Microsoft YaHei', 'SimHei', 'Arial Black';
                font-weight: 900;
            }
            QGroupBox#guideGroup::title {
                subcontrol-origin: margin;
                left: 18px;
                padding: 0 14px;
                color: #C44B4F;
                font-weight: 900;
                background-color: #F2EDE4;
            }
        """)
        guide_layout = QVBoxLayout(quick_guide)
        guide_layout.setContentsMargins(15, 25, 15, 15)
        guide_layout.setSpacing(10)

        # 使用流程 - 红色标题条
        steps_title = QLabel("▸ 使用流程")
        steps_title.setFont(QFont("Microsoft YaHei", 11, QFont.Black))
        steps_title.setStyleSheet("color: #FFFFFF; background-color: #C44B4F; padding: 8px 14px; font-weight: 900; border: 2px solid #2B2B2B;")
        guide_layout.addWidget(steps_title)

        steps_text = QLabel(
            "<div style='line-height: 2.4; font-size: 10.5pt; color: #2B2B2B;'>"
            "<div style='background: #F2EDE4; padding: 6px 10px; margin: 3px 0 3px 0px; border-left: 6px solid #C44B4F;'>"
            "<b style='color: #FFF; background: #C44B4F; padding: 2px 8px; font-size: 13pt;'>1</b>　→ 等待模型加载完成"
            "</div>"
            "<div style='background: #F2EDE4; padding: 6px 10px; margin: 3px 0 3px 12px; border-left: 6px solid #2B2B2B;'>"
            "<b style='color: #FFF; background: #8A8580; padding: 2px 8px; font-size: 13pt;'>2</b>　→ 点击「开始录音」按钮"
            "</div>"
            "<div style='background: #F2EDE4; padding: 6px 10px; margin: 3px 0 3px 24px; border-left: 6px solid #C44B4F;'>"
            "<b style='color: #FFF; background: #C44B4F; padding: 2px 8px; font-size: 13pt;'>3</b>　→ 说出您的感受（3-30秒）"
            "</div>"
            "<div style='background: #F2EDE4; padding: 6px 10px; margin: 3px 0 3px 12px; border-left: 6px solid #2B2B2B;'>"
            "<b style='color: #FFF; background: #8A8580; padding: 2px 8px; font-size: 13pt;'>4</b>　→ 点击「停止录音」按钮"
            "</div>"
            "<div style='background: #F2EDE4; padding: 6px 10px; margin: 3px 0 3px 0px; border-left: 6px solid #C44B4F;'>"
            "<b style='color: #FFF; background: #C44B4F; padding: 2px 8px; font-size: 13pt;'>5</b>　→ 查看情绪分析结果"
            "</div>"
            "</div>"
        )
        steps_text.setWordWrap(True)
        steps_text.setTextFormat(Qt.RichText)
        guide_layout.addWidget(steps_text)

        # 功能说明
        features_title = QLabel("▸ 功能说明")
        features_title.setFont(QFont("Microsoft YaHei", 11, QFont.Black))
        features_title.setStyleSheet("color: #FFFFFF; background-color: #C44B4F; padding: 8px 14px; font-weight: 900; margin-top: 10px; border: 2px solid #2B2B2B;")
        guide_layout.addWidget(features_title)

        features_text = QLabel(
            "<div style='line-height: 1.9; font-size: 10pt; color: #2B2B2B;'>"
            "\u25aa <b>多模型支持：</b>可切换 Large/Base/Seed 三种模型<br>"
            "\u25aa <b>7种情绪识别：</b>平静、开心、惊讶、悲伤、愤怒、恐惧、厌恶<br>"
            "\u25aa <b>复合情绪检测：</b>自动识别焦虑、挫败等复合情绪<br>"
            "\u25aa <b>情绪稳定度：</b>0-10分，分数越低越稳定<br>"
            "\u25aa <b>历史报告：</b>查看历史记录与情绪趋势图<br>"
            "\u25aa <b>数据管理：</b>清理录音、缓存等数据文件<br>"
            "\u25aa <b>便携模式：</b>所有数据保存在程序文件夹"
            "</div>"
        )
        features_text.setWordWrap(True)
        features_text.setTextFormat(Qt.RichText)
        guide_layout.addWidget(features_text)

        # 提示
        tip_label = QLabel(
            "<div style='background: #F2EDE4; padding: 10px 12px; margin-top: 6px; font-size: 10pt; color: #2B2B2B; border: 2px solid #2B2B2B;'>"
            "<b>小贴士：</b>在安静的环境下录音，效果会更好"
            "</div>"
        )
        tip_label.setWordWrap(True)
        tip_label.setTextFormat(Qt.RichText)
        guide_layout.addWidget(tip_label)

        right_layout.addWidget(quick_guide)

        content_splitter.addWidget(right_panel)

        content_splitter.setStretchFactor(0, 7)
        content_splitter.setStretchFactor(1, 3)
        content_splitter.setSizes([840, 360])

        layout.addWidget(content_splitter)
        self.tab_widget.addTab(realtime_widget, "▸ 实时检测")

    @exception_safe()
    def create_history_tab(self):
        history_widget = QWidget()
        layout = QVBoxLayout(history_widget)
        layout.setSpacing(12)

        history_btn_layout = QHBoxLayout()
        history_btn_layout.setSpacing(10)

        self.history_refresh_btn = QPushButton("刷新列表")
        self.history_refresh_btn.setFont(QFont("Microsoft YaHei", 11))
        self.history_refresh_btn.setMinimumHeight(38)
        self.history_refresh_btn.setMinimumWidth(120)
        self.history_refresh_btn.setCursor(Qt.PointingHandCursor)
        self.history_refresh_btn.clicked.connect(self._on_history_refresh_clicked)
        history_btn_layout.addWidget(self.history_refresh_btn)

        history_btn_layout.addStretch()

        record_count_label = QLabel("")
        record_count_label.setFont(QFont("Microsoft YaHei", 10))
        record_count_label.setObjectName("countLabel")
        self.record_count_label = record_count_label
        history_btn_layout.addWidget(record_count_label)

        layout.addLayout(history_btn_layout)

        # P1 新增：数据导出工具栏
        self.export_toolbar = ExportToolbar()
        self.export_toolbar.export_csv_clicked.connect(self._on_export_csv)
        self.export_toolbar.export_json_clicked.connect(self._on_export_json)
        layout.addWidget(self.export_toolbar)

        splitter = QSplitter(Qt.Vertical)
        splitter.setObjectName("historySplitter")

        list_group = QGroupBox("■ 历史记录")
        list_group.setFont(QFont("Microsoft YaHei", 12, QFont.Black))
        list_group.setObjectName("historyGroup")
        list_group.setStyleSheet("""
            QGroupBox#historyGroup {
                border: 3px solid #2B2B2B;
                border-top: 6px solid #C44B4F;
                border-radius: 0px;
                margin-top: 14px;
                padding-top: 22px;
                background-color: #F2EDE4;
            }
            QGroupBox#historyGroup::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 2px 14px;
                color: #C44B4F;
                font-weight: 900;
            }
        """)
        list_layout = QVBoxLayout(list_group)
        list_layout.setContentsMargins(15, 25, 15, 15)
        self.history_list = QListWidget()
        self.history_list.setFont(QFont("Microsoft YaHei", 11))
        self.history_list.setObjectName("historyList")
        self.history_list.setMinimumHeight(180)
        self.history_list.setMaximumHeight(280)
        self.history_list.setSpacing(2)
        self.history_list.setSelectionMode(QListWidget.NoSelection)
        list_layout.addWidget(self.history_list)
        splitter.addWidget(list_group)

        plot_group = QGroupBox("■ 情绪趋势图")
        plot_group.setFont(QFont("Microsoft YaHei", 12, QFont.Black))
        plot_group.setObjectName("plotGroup")
        plot_group.setStyleSheet("""
            QGroupBox#plotGroup {
                border: 3px solid #2B2B2B;
                border-top: 6px solid #C44B4F;
                border-radius: 0px;
                margin-top: 14px;
                padding-top: 22px;
                background-color: #F2EDE4;
            }
            QGroupBox#plotGroup::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 2px 14px;
                color: #C44B4F;
                font-weight: 900;
            }
        """)
        plot_layout = QVBoxLayout(plot_group)
        plot_layout.setContentsMargins(15, 25, 15, 10)

        if MATPLOTLIB_AVAILABLE:
            self.canvas = MplCanvas(self, width=10, height=7, dpi=100)
            self.canvas.setMinimumHeight(350)
            plot_layout.addWidget(self.canvas)
        else:
            plot_placeholder = QLabel(
                "绘图库未安装\n\n趋势图功能暂不可用"
            )
            plot_placeholder.setAlignment(Qt.AlignCenter)
            plot_placeholder.setFont(QFont("Microsoft YaHei", 12))
            plot_placeholder.setStyleSheet("color: #8C8680; padding: 40px;")
            plot_layout.addWidget(plot_placeholder)

        splitter.addWidget(plot_group)
        splitter.setSizes([250, 550])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)
        self.tab_widget.addTab(history_widget, "▸ 历史报告")

    @exception_safe()
    def create_bottom_buttons(self, parent_layout):
        bottom_container = QVBoxLayout()
        bottom_container.setSpacing(8)

        # 构成主义底部操作条：红色底板（象征"行动带"）+ 粗黑边框
        btn_frame = QFrame()
        btn_frame.setObjectName("bottomBtnFrame")
        btn_frame.setMinimumHeight(42)
        btn_frame.setStyleSheet("""
            QFrame#bottomBtnFrame {
                background-color: #C44B4F;
                border: 4px solid #2B2B2B;
                border-radius: 0px;
            }
        """)
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setSpacing(15)
        btn_layout.setContentsMargins(20, 5, 20, 5)

        data_btn = QPushButton("数据管理")
        data_btn.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        data_btn.setMinimumHeight(48)
        data_btn.setMinimumWidth(150)
        data_btn.setObjectName("dataBtn")
        data_btn.clicked.connect(self.show_data_manager)
        data_btn.setCursor(Qt.PointingHandCursor)
        data_btn.setStyleSheet("""
            QPushButton#dataBtn {
                background-color: transparent;
                color: #FFFFFF;
                border: 2px solid #FFFFFF;
                border-radius: 0px;
                padding: 12px 28px;
                font-weight: bold;
                font-family: 'Microsoft YaHei', 'SimHei', 'Arial Black';
            }
            QPushButton#dataBtn:hover {
                background-color: #2B2B2B;
                color: #FFFFFF;
                border-color: #FFFFFF;
            }
        """)
        btn_layout.addWidget(data_btn)

        btn_layout.addStretch()

        bottom_container.addWidget(btn_frame)

        current_model = load_model_config()
        self.credit_label = QLabel(
            f"Powered by {current_model} (ModelScope/达摩院, Apache-2.0) | "
            "便携模式 - 数据保存在程序目录 | 仅供个人非商用参考使用"
        )
        self.credit_label.setAlignment(Qt.AlignCenter)
        self.credit_label.setFont(QFont("Microsoft YaHei", 8, QFont.Bold))
        self.credit_label.setObjectName("creditLabel")
        self.credit_label.setWordWrap(True)
        bottom_container.addWidget(self.credit_label)

        parent_layout.addLayout(bottom_container)

    @exception_safe()
    def setup_styles(self):
        qss = """
        QMainWindow {
            background-color: #F2EDE4;
        }

        #headerFrame {
            background-color: #F2EDE4;
            border: 3px solid #2B2B2B;
            border-radius: 0px;
        }

        #headerTitle {
            color: #2B2B2B;
            font-family: "Microsoft YaHei", "SimHei", "Arial Black";
            font-weight: 900;
            font-style: oblique;
        }

        #statusReady {
            color: #FFFFFF;
            background-color: #2B2B2B;
            padding: 8px 18px;
            border-radius: 0px;
            border: 2px solid #2B2B2B;
            font-weight: bold;
        }

        #statusLoading {
            color: #FFFFFF;
            background-color: #8A8580;
            padding: 8px 18px;
            border-radius: 0px;
            border: 2px solid #2B2B2B;
            font-weight: bold;
        }

        QMenuBar {
            background-color: #F2EDE4;
            border-bottom: 3px solid #2B2B2B;
            font-family: "Microsoft YaHei", "SimHei", "Arial Black";
            font-size: 11px;
            font-weight: bold;
        }

        QMenuBar::item {
            padding: 10px 18px;
            background-color: transparent;
            border-radius: 0px;
        }

        QMenuBar::item:selected {
            background-color: #2B2B2B;
            color: #FFFFFF;
        }

        QMenu {
            background-color: #F2EDE4;
            border: 2px solid #2B2B2B;
            font-family: "Microsoft YaHei", "SimHei", "Arial Black";
            font-size: 11px;
            font-weight: bold;
            border-radius: 0px;
            padding: 6px;
        }

        QMenu::item {
            padding: 10px 25px;
            border-radius: 0px;
        }

        QMenu::item:selected {
            background-color: #C44B4F;
            color: #FFFFFF;
        }

        QTabWidget::pane {
            border: none;
            border-radius: 0px;
            background-color: transparent;
        }

        QTabBar::tab {
            background-color: #E8E3DA;
            padding: 14px 36px;
            margin-right: 6px;
            border-top-left-radius: 0px;
            border-top-right-radius: 0px;
            font-weight: 900;
            font-size: 14px;
            font-family: "Microsoft YaHei", "SimHei", "Arial Black";
            color: #2B2B2B;
            border: 3px solid #2B2B2B;
            border-bottom: none;
        }

        QTabBar::tab:selected {
            background-color: #F2EDE4;
            border-bottom: 4px solid #C44B4F;
            color: #C44B4F;
        }

        QTabBar::tab:hover:!selected {
            background-color: #C44B4F;
            color: #FFFFFF;
        }

        QGroupBox {
            border: 3px solid #2B2B2B;
            border-radius: 0px;
            margin-top: 14px;
            padding-top: 22px;
            background-color: #F2EDE4;
            font-family: "Microsoft YaHei", "SimHei", "Arial Black";
            font-weight: bold;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 20px;
            padding: 2px 14px;
            color: #C44B4F;
            font-weight: 900;
            background-color: #F2EDE4;
        }

        #recordBtn {
            background-color: #8A8580;
            color: white;
            border: 3px solid #2B2B2B;
            border-radius: 0px;
            padding: 20px;
            font-family: "Microsoft YaHei", "SimHei", "Arial Black";
            font-weight: bold;
        }

        #recordBtn:enabled {
            background-color: #C44B4F;
        }

        #recordBtn:enabled:hover {
            background-color: #A33B3F;
        }

        #recordBtn:enabled:pressed {
            background-color: #8B2F32;
        }

        #recordBtn[isRecording="true"] {
            background-color: #2B2B2B;
        }

        #recordBtn[isRecording="true"]:hover {
            background-color: #1A1A1A;
        }

        #durationLabel {
            background-color: #F2EDE4;
            border-radius: 0px;
            padding: 10px 20px;
            color: #2B2B2B;
            border: 3px solid #2B2B2B;
            font-weight: bold;
        }

        #recordProgress {
            border: 3px solid #2B2B2B;
            border-radius: 0px;
            background-color: #E8E3DA;
        }

        #recordProgress::chunk {
            background-color: #C44B4F;
            border-radius: 0px;
        }

        #scoreCard {
            background-color: rgba(242, 237, 228, 240);
            border-radius: 0px;
            border: 3px solid #2B2B2B;
        }

        #scoreCard:hover {
            border-color: #C44B4F;
        }

        #cardTitle {
            color: #2B2B2B;
            font-weight: bold;
            font-family: "Microsoft YaHei", "SimHei", "Arial Black";
        }

        #cardValue {
            color: #2B2B2B;
            font-family: "Consolas", "Courier New", monospace;
            font-weight: bold;
        }

        #cardSub {
            color: #8A8580;
            font-weight: bold;
        }

        #warningLabel {
            border-radius: 0px;
            padding: 12px 20px;
            border: 3px solid #2B2B2B;
            background-color: #F2EDE4;
            font-weight: 900;
        }

        QProgressBar {
            border: 2px solid #2B2B2B;
            border-radius: 0px;
            background-color: #E8E3DA;
            text-align: center;
            font-weight: bold;
            font-size: 10pt;
            font-family: "Microsoft YaHei", "SimHei", "Arial Black";
        }

        QProgressBar::chunk {
            border-radius: 0px;
            margin: 0px;
        }

        #probBar_平静::chunk { background-color: #C44B4F; }
        #probBar_开心::chunk { background-color: #C44B4F; }
        #probBar_惊讶::chunk { background-color: #C44B4F; }
        #probBar_悲伤::chunk { background-color: #C44B4F; }
        #probBar_愤怒::chunk { background-color: #C44B4F; }
        #probBar_恐惧::chunk { background-color: #C44B4F; }
        #probBar_厌恶::chunk { background-color: #C44B4F; }

        #historyList {
            border: 3px solid #2B2B2B;
            border-radius: 0px;
            background-color: #F2EDE4;
            padding: 8px;
            font-family: "Microsoft YaHei", "SimHei", "Arial Black";
            font-weight: bold;
        }

        #historyList::item {
            padding: 12px 15px;
            border-bottom: 2px solid #2B2B2B;
            border-left: 5px solid transparent;
            border-radius: 0px;
            margin: 2px 0;
        }

        #historyList::item:selected {
            background-color: #2B2B2B;
            color: white;
            border-left: 5px solid #C44B4F;
        }

        #historyList::item:hover:!selected {
            background-color: #E8E3DA;
            color: #2B2B2B;
            border-left: 5px solid #8A8580;
        }

        #countLabel {
            color: #2B2B2B;
            padding: 5px 10px;
            font-weight: bold;
        }

        #dangerBtn {
            background-color: #C44B4F;
            color: white;
            border: 3px solid #2B2B2B;
            border-radius: 0px;
            padding: 8px 20px;
            font-weight: bold;
        }

        #dangerBtn:hover {
            background-color: #A33B3F;
        }

        QScrollArea {
            border: none;
            background-color: transparent;
        }

        QScrollBar:vertical {
            border: none;
            background: #E8E3DA;
            width: 10px;
            border-radius: 0px;
        }

        QScrollBar::handle:vertical {
            background: #2B2B2B;
            border-radius: 0px;
            min-height: 30px;
        }

        QScrollBar::handle:vertical:hover {
            background: #C44B4F;
        }

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
        }

        #creditLabel {
            color: #8A8580;
            padding: 5px;
            font-weight: bold;
        }

        QPushButton {
            outline: none;
            font-family: "Microsoft YaHei", "SimHei", "Arial Black";
            font-weight: bold;
        }

        QPushButton:disabled {
            background-color: #E8E3DA !important;
            color: #8A8580 !important;
            cursor: not-allowed;
        }

        QSplitter::handle {
            background-color: #2B2B2B;
            width: 3px;
        }

        QSplitter::handle:hover {
            background-color: #C44B4F;
            border-radius: 0px;
        }

        QLabel {
            font-family: "Microsoft YaHei", "SimHei", "Arial Black";
        }
        """
        self.setStyleSheet(qss)

    def toggle_recording(self, checked=False):
        try:
            self.append_log("检测到录音按钮点击")
            logger.info("录音按钮被点击")

            if not self.recognizer:
                self.append_log("错误：识别器未初始化")
                QMessageBox.warning(self, "错误", "识别器未初始化，请重启程序")
                return

            if not self.recognizer.is_ready():
                self.append_log("模型未就绪")
                QMessageBox.warning(self, "请稍候", "模型还在加载中，请等待加载完成后再开始录音")
                return

            if self.is_analyzing:
                QMessageBox.information(self, "请稍候", "正在分析中，请等待当前分析完成")
                return

            if not self.is_recording:
                self.append_log("准备开始录音...")
                self.start_recording_action()
            else:
                self.append_log("准备停止录音...")
                self.stop_recording_action()
        except Exception as e:
            logger.error(f"录音切换异常: {str(e)}", exc_info=True)
            self.append_log(f"录音操作异常: {str(e)}")
            QMessageBox.critical(self, "错误", f"录音操作失败:\n{str(e)}")
            self._reset_recording_ui()

    @exception_safe()
    def _reset_recording_ui(self):
        self.is_recording = False
        self.is_analyzing = False
        self.record_btn.setText("● 点击开始录音")
        self.record_btn.setProperty("isRecording", "false")
        self.record_btn.style().unpolish(self.record_btn)
        self.record_btn.style().polish(self.record_btn)
        if self.recognizer and self.recognizer.is_ready():
            self.record_btn.setEnabled(True)
        self.record_progress.setRange(0, 100)
        self.record_progress.setValue(0)
        self._set_buttons_enabled(True)

    def start_recording_action(self):
        try:
            self.append_log("正在初始化录音...")
            logger.info("开始执行 start_recording_action")

            if self.is_recording:
                self.append_log("录音已在进行中")
                return

            self.record_btn.setEnabled(False)

            recordings_dir = get_recordings_dir()
            self.append_log(f"录音目录: {recordings_dir}")
            if not os.path.exists(recordings_dir):
                os.makedirs(recordings_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_audio_path = os.path.join(recordings_dir, f"recording_{timestamp}.wav")

            test_file = os.path.join(recordings_dir, '.write_test')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except (OSError, PermissionError, IOError) as e:
                QMessageBox.critical(self, "错误", f"录音目录不可写:\n{str(e)}\n\n请检查程序文件夹权限")
                self._reset_recording_ui()
                return

            self.append_log("正在检测音频设备...")
            try:
                import pyaudio
                p = pyaudio.PyAudio()
                device_count = p.get_device_count()
                self.append_log(f"找到 {device_count} 个音频设备")
                input_devices = []
                for i in range(device_count):
                    dev_info = p.get_device_info_by_index(i)
                    if dev_info.get('maxInputChannels', 0) > 0:
                        input_devices.append(dev_info.get('name', f'Device {i}'))
                if not input_devices:
                    p.terminate()
                    raise Exception("未找到可用的麦克风输入设备，请检查麦克风是否连接")
                self.append_log(f"可用麦克风: {', '.join(input_devices[:2])}")
                p.terminate()
            except ImportError:
                QMessageBox.critical(self, "错误", "PyAudio库未正确安装，请重新安装依赖")
                self._reset_recording_ui()
                return
            except Exception as e:
                if "No Default Input Device Available" in str(e):
                    QMessageBox.critical(self, "错误", "未找到默认麦克风，请连接麦克风后重试")
                else:
                    QMessageBox.critical(self, "错误", f"音频设备检测失败:\n{str(e)}")
                self._reset_recording_ui()
                return

            self.append_log("正在创建录音设备...")
            self.recorder = AudioRecorder()

            self.append_log("正在创建录音线程...")
            self.recording_thread = RecordingThread(self.recorder, self.current_audio_path)
            self.recording_thread.duration_updated.connect(self.on_duration_updated)
            self.recording_thread.recording_finished.connect(self.on_recording_finished)
            self.recording_thread.recording_error.connect(self.on_recording_error)

            self.is_recording = True
            self.is_analyzing = False
            self.record_btn.setText("点击停止录音")
            self.record_btn.setProperty("isRecording", "true")
            self.record_btn.style().unpolish(self.record_btn)
            self.record_btn.style().polish(self.record_btn)
            self.record_progress.setRange(0, 0)
            self.duration_display.setText("00:00")
            self.status_label.setText("● 正在录音...")
            self.status_label.setStyleSheet("color: #FFFFFF; background-color: #C44B4F; padding: 8px 18px; border-radius: 0px; border: 2px solid #2B2B2B; font-weight: bold;")

            self.recording_thread.start()

            import time
            time.sleep(0.1)
            if not self.recording_thread.isRunning():
                raise Exception("录音线程启动失败")

            self.record_btn.setEnabled(True)
            self.append_log(f"录音已开始")
            logger.info(f"开始录音: {self.current_audio_path}")
        except Exception as e:
            logger.error(f"启动录音失败: {str(e)}", exc_info=True)
            self.append_log(f"启动录音失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"启动录音失败:\n{str(e)}")
            self._reset_recording_ui()

    def stop_recording_action(self):
        try:
            self.append_log("请求停止录音...")
            logger.info("stop_recording_action 被调用")

            if not self.is_recording:
                self.append_log("当前没有在录音")
                return

            self.is_recording = False
            self.record_btn.setText("正在保存...")
            self.record_btn.setEnabled(False)
            self.status_label.setText("● 处理中...")
            self.status_label.setStyleSheet("color: #FFFFFF; background-color: #8A8580; padding: 8px 18px; border-radius: 0px; border: 2px solid #2B2B2B; font-weight: bold;")
            self.record_progress.setRange(0, 100)
            self.record_progress.setValue(50)

            if self.recording_thread and self.recording_thread.isRunning():
                self.append_log("正在停止录音...")
                self.recording_thread.stop()
                logger.info("已发送停止信号")
                self.append_log("停止信号已发送，正在保存录音...")
            else:
                self.append_log("录音线程未运行，直接保存...")
                if hasattr(self, 'recorder') and self.recorder:
                    try:
                        output_path = self.recorder.stop_recording()
                        if output_path:
                            self.on_recording_finished(output_path)
                        else:
                            self.append_log("停止录音失败")
                            self._reset_recording_ui()
                    except Exception as e2:
                        logger.error(f"强制停止异常: {e2}")
                        self._reset_recording_ui()
        except Exception as e:
            logger.error(f"停止录音异常: {str(e)}", exc_info=True)
            self.append_log(f"■ 停止录音异常: {str(e)}")
            QMessageBox.critical(self, "错误", f"停止录音失败:\n{str(e)}")
            self._reset_recording_ui()

    @exception_safe()
    def on_duration_updated(self, duration):
        minutes = int(duration) // 60
        seconds = int(duration) % 60
        self.duration_display.setText(f"{minutes:02d}:{seconds:02d}")

        # P0 新增：实时录音质量反馈
        try:
            volume = self.recorder.get_volume_level() if self.recorder else 0.0
            if duration < 1.0:
                self.quality_feedback.setText("正在录音...")
                self.quality_feedback.setStyleSheet(
                    "background-color: #F2EDE4; border: 2px solid #2B2B2B; "
                    "padding: 4px; color: #8A8580; font-weight: bold;"
                )
            elif volume < 0.02:
                self.quality_feedback.setText("⚠ 音量过低 — 请靠近麦克风")
                self.quality_feedback.setStyleSheet(
                    "background-color: #F2EDE4; border: 2px solid #C44B4F; "
                    "padding: 4px; color: #C44B4F; font-weight: bold;"
                )
            elif volume > 0.95:
                self.quality_feedback.setText("⚠ 音量过高 — 可能爆音")
                self.quality_feedback.setStyleSheet(
                    "background-color: #F2EDE4; border: 2px solid #C44B4F; "
                    "padding: 4px; color: #C44B4F; font-weight: bold;"
                )
            else:
                bar_len = int(volume * 12)
                bar = "█" * bar_len + "░" * (12 - bar_len)
                self.quality_feedback.setText(f"✓ 音量正常 {bar}")
                self.quality_feedback.setStyleSheet(
                    "background-color: #F2EDE4; border: 2px solid #2B2B2B; "
                    "padding: 4px; color: #2B2B2B; font-weight: bold;"
                )
        except Exception:
            pass

    @exception_safe()
    def on_recording_finished(self, output_path):
        self.is_recording = False
        self.record_btn.setText("● 点击开始录音")
        self.record_btn.setProperty("isRecording", "false")
        self.record_btn.style().unpolish(self.record_btn)
        self.record_btn.style().polish(self.record_btn)
        self.record_progress.setRange(0, 100)
        self.record_progress.setValue(100)

        try:
            duration = self.recorder.get_duration()
            self.append_log(f"录音完成，时长: {duration:.1f}秒")
            logger.info(f"录音完成: {duration:.1f}秒, {output_path}")

            if duration < 1.0:
                self.record_progress.setRange(0, 100)
                self.record_progress.setValue(0)
                try:
                    if os.path.exists(output_path):
                        safe_remove_file(output_path)
                except Exception:
                    pass
                QMessageBox.warning(self, "提示", "录音时长过短，请重新录制（至少1秒）")
                self.append_log("录音时长过短，已取消分析")
                self._reset_recording_ui()
                self.status_label.setText("● 准备就绪")
                self.status_label.setStyleSheet("color: #FFFFFF; background-color: #2B2B2B; padding: 8px 18px; border-radius: 0px; border: 2px solid #2B2B2B; font-weight: bold;")
                return

            if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
                QMessageBox.warning(self, "错误", "录音文件无效，请重新录制")
                self._reset_recording_ui()
                return

            self.is_analyzing = True
            self.start_analysis(output_path)
        except Exception as e:
            logger.error(f"录音完成处理异常: {str(e)}", exc_info=True)
            self.append_log(f"处理录音文件异常: {str(e)}")
            QMessageBox.critical(self, "错误", f"处理录音文件失败:\n{str(e)}")
            self._reset_recording_ui()

    @exception_safe()
    def on_recording_error(self, error_msg):
        self._reset_recording_ui()
        self.append_log(f"录音错误: {error_msg}")
        logger.error(f"录音错误: {error_msg}")
        QMessageBox.critical(self, "录音错误", f"录音过程发生错误:\n{error_msg}\n\n请检查麦克风是否正常连接，以及程序是否有录音权限")

    @exception_safe()
    def start_analysis(self, audio_path):
        try:
            if not self.recognizer or not self.recognizer.is_ready():
                QMessageBox.warning(self, "请稍候", "模型还未加载完成，请等待")
                self._reset_recording_ui()
                return

            if not os.path.exists(audio_path):
                QMessageBox.warning(self, "错误", "音频文件不存在")
                self._reset_recording_ui()
                return

            self.append_log("正在分析情绪...")
            logger.info("开始情绪分析")
            self.record_progress.setRange(0, 0)
            self.record_btn.setEnabled(False)

            self.analysis_thread = AnalysisThread(self.recognizer, audio_path)
            self.analysis_thread.analysis_finished.connect(self.on_analysis_finished)
            self.analysis_thread.analysis_error.connect(self.on_analysis_error)
            self.analysis_thread.start()
        except Exception as e:
            logger.error(f"启动分析异常: {str(e)}", exc_info=True)
            self.append_log(f"启动分析失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"启动分析失败:\n{str(e)}")
            self._reset_recording_ui()

    @exception_safe()
    def on_analysis_error(self, error_msg):
        self.record_progress.setRange(0, 100)
        self.record_progress.setValue(0)
        self.is_analyzing = False
        self._reset_recording_ui()
        self.append_log(f"分析异常: {error_msg}")
        logger.error(f"分析异常: {error_msg}")
        self.status_label.setText("● 分析失败")
        self.status_label.setStyleSheet("color: #FFFFFF; background-color: #C44B4F; padding: 8px 18px; border-radius: 0px; border: 2px solid #2B2B2B; font-weight: bold;")
        QMessageBox.critical(self, "分析错误", f"情绪分析过程发生错误:\n{error_msg}")

    @exception_safe()
    def _set_buttons_enabled(self, enabled):
        self.record_btn.setEnabled(enabled and self.recognizer and self.recognizer.is_ready())

    @exception_safe()
    def on_analysis_finished(self, result):
        self.record_progress.setRange(0, 100)
        self.record_progress.setValue(100)
        self.is_analyzing = False
        self._set_buttons_enabled(True)

        try:
            if not isinstance(result, dict):
                QMessageBox.warning(self, "错误", "分析结果格式异常")
                return

            if result.get('success', False):
                score = result.get('情绪稳定度分数', 0.0)
                level = result.get('情绪状态等级', '未知')
                color = result.get('等级颜色', '#95A5A6')
                probs = result.get('所有情绪概率', {})
                main_emotion = result.get('主要情绪', '未知')
                confidence = result.get('置信度', 0.0)
                advice = result.get('调节建议', '')
                mixed_emotions = result.get('混合情绪', [])

                result['audio_file'] = self.current_audio_path

                # P0 新增：添加实验元数据（版本 + 模型名）
                from main import APP_VERSION, ALGORITHM_VERSION
                result['algorithm_version'] = ALGORITHM_VERSION
                result['app_version'] = APP_VERSION
                result['model_name'] = self.recognizer.model_name if self.recognizer else ''

                # P0 新增：计算音频质量
                from audio_quality import AudioQualityAnalyzer
                try:
                    analyzer = AudioQualityAnalyzer()
                    audio_quality = analyzer.analyze(self.current_audio_path)
                    if audio_quality:
                        result['audio_quality'] = audio_quality
                        result['assessment_reliability'] = (
                            "高" if audio_quality.get('quality_score', 0) >= 0.6 else
                            "中" if audio_quality.get('quality_score', 0) >= 0.4 else "低"
                        )
                except Exception:
                    pass

                self.score_card.set_value(f"{score:.1f}", color, "分")
                self.level_card.set_value(level, color)
                self.emotion_card.set_value(main_emotion, "#3498db", f"置信度 {confidence:.1%}")

                # P0 新增：更新集成化评估报告卡片
                self.result_card_widget.update_result(result)
                self.result_card_widget.show()

                # P1 新增：更新 VAD 维度雷达图
                self.radar_chart.update_values(
                    valence=result.get('valence_score', 0.0),
                    arousal=result.get('arousal_score', 0.0),
                    dominance=result.get('dominance_score', 0.0),
                    negative_load=result.get('negative_load', 0.0),
                    uncertainty=result.get('emotional_uncertainty', 0.0)
                )
                self.radar_chart.show()

                # 复合情绪卡片展示
                compound_emotion = result.get('复合情绪', '')
                compound_detail = result.get('复合情绪详情', None)
                if compound_emotion:
                    self.compound_card.set_value(compound_emotion, "#9B59B6", compound_detail.get('desc', '')[:20] if compound_detail else "")
                else:
                    self.compound_card.set_value("未检测到", "#95A5A6", "情绪状态较单一")

                from emotion_recognizer import STABILITY_LEVELS
                border_color = color
                bg_color = border_color + "15"

                # 构成主义：始终使用炭黑边框，不以情绪颜色改变结构
                self.result_group.setStyleSheet(f"""
                    QGroupBox {{
                        border: 3px solid #2B2B2B;
                        border-radius: 0px;
                        margin-top: 12px;
                        padding-top: 20px;
                        background-color: #F2EDE4;
                    }}
                    QGroupBox::title {{
                        subcontrol-origin: margin;
                        left: 20px;
                        padding: 0 12px;
                        color: #2B2B2B;
                        font-weight: bold;
                    }}
                """)

                for emotion, bar in self.prob_bars.items():
                    try:
                        prob = float(probs.get(emotion, 0.0))
                        pct = int(prob * 100)
                        bar.setValue(pct)
                        if emotion in self.prob_labels:
                            self.prob_labels[emotion].setText(f"{pct}%")
                    except (TypeError, ValueError):
                        bar.setValue(0)
                        if emotion in self.prob_labels:
                            self.prob_labels[emotion].setText("0%")


                result_copy = dict(result)
                result_copy['anxiety_score'] = score
                self.history_manager.add_record(result_copy)

                if score >= 6.5:
                    self.warning_label.show()
                    if score >= 8:
                        self.warning_label.setText("情绪波动较大！建议立即进行深呼吸放松，必要时寻求亲友陪伴或专业帮助")
                        self.warning_label.setStyleSheet("""
                            background-color: #F2EDE4;
                            color: #C44B4F;
                            border: 3px solid #C44B4F;
                            border-radius: 0px;
                            padding: 12px 20px;
                            font-weight: bold;
                        """)
                        self.status_label.setText("● 情绪不稳定")
                        self.status_label.setStyleSheet("color: #FFFFFF; background-color: #C44B4F; padding: 8px 18px; border-radius: 0px; border: 2px solid #2B2B2B; font-weight: bold;")
                    else:
                        self.warning_label.setText("情绪存在一定波动，建议适当休息放松")
                        self.warning_label.setStyleSheet("""
                            background-color: #F2EDE4;
                            color: #8A8580;
                            border: 3px solid #8A8580;
                            border-radius: 0px;
                            padding: 12px 20px;
                            font-weight: bold;
                        """)
                        self.status_label.setText("● 情绪波动")
                        self.status_label.setStyleSheet("color: #FFFFFF; background-color: #8A8580; padding: 8px 18px; border-radius: 0px; border: 2px solid #2B2B2B; font-weight: bold;")
                else:
                    self.warning_label.hide()
                    self.status_label.setText("● 情绪状态良好")
                    self.status_label.setStyleSheet("color: #FFFFFF; background-color: #2B2B2B; padding: 8px 18px; border-radius: 0px; border: 2px solid #2B2B2B; font-weight: bold;")

                self.append_log(f"分析完成 - 稳定度分数: {score:.1f}/10, 情绪状态: {level}, 主要情绪: {main_emotion}" + (f", 复合情绪: {compound_emotion}" if compound_emotion else ""))
                logger.info(f"分析完成 - 稳定度: {score:.1f}, 状态: {level}, 情绪: {main_emotion}" + (f", 复合情绪: {compound_emotion}" if compound_emotion else ""))
            else:
                error_msg = result.get('error', '未知错误')
                QMessageBox.warning(self, "分析错误", f"情绪分析失败:\n{error_msg}")
                self.append_log(f"分析错误: {error_msg}")
                logger.error(f"分析失败: {error_msg}")
                self.status_label.setText("● 分析失败")
                self.status_label.setStyleSheet("color: #FFFFFF; background-color: #C44B4F; padding: 8px 18px; border-radius: 0px; border: 2px solid #2B2B2B; font-weight: bold;")
        except Exception as e:
            logger.error(f"处理分析结果异常: {str(e)}", exc_info=True)
            self.append_log(f"处理分析结果异常: {str(e)}")
            QMessageBox.critical(self, "错误", f"处理分析结果失败:\n{str(e)}")
            self.status_label.setText("● 处理错误")
            self.status_label.setStyleSheet("color: #FFFFFF; background-color: #C44B4F; padding: 8px 18px; border-radius: 0px; border: 2px solid #2B2B2B; font-weight: bold;")

    @exception_safe()
    def on_tab_changed(self, index):
        try:
            tab_text = self.tab_widget.tabText(index)
            if "历史报告" in tab_text:
                self.refresh_history()
        except Exception as e:
            logger.error(f"切换标签页异常: {str(e)}", exc_info=True)

    def _on_history_refresh_clicked(self):
        """历史记录刷新按钮点击（带动画效果）"""
        try:
            self.history_refresh_btn.setText("刷新中...")
            self.history_refresh_btn.setEnabled(False)
            # 列表区域淡出效果
            self.history_list.setStyleSheet("opacity: 0.3; border: 2px solid #2B2B2B; border-radius: 0px; background-color: #E8E3DA; padding: 8px;")
            from PyQt5.QtCore import QTimer
            def _do_refresh():
                self.refresh_history()
                self.history_list.setStyleSheet("")
                self.history_refresh_btn.setText("刷新列表")
                self.history_refresh_btn.setEnabled(True)
            QTimer.singleShot(250, _do_refresh)
        except Exception as e:
            logger.error(f"刷新动画异常: {str(e)}")
            self.refresh_history()
            self.history_refresh_btn.setText("刷新列表")
            self.history_refresh_btn.setEnabled(True)

    @exception_safe()
    def _on_export_csv(self):
        """导出历史记录为 CSV"""
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 CSV 研究数据", "emotion_data.csv",
            "CSV 文件 (*.csv);;所有文件 (*)"
        )
        if path:
            success = self.history_manager.export_records_csv(path)
            if success:
                QMessageBox.information(self, "导出成功",
                    f"数据已导出到:\n{path}\n\n格式: UTF-8 CSV (Excel/SPSS兼容)")
            else:
                QMessageBox.warning(self, "导出失败", "没有可导出的记录或导出过程出错")

    @exception_safe()
    def _on_export_json(self):
        """导出历史记录为 JSON 研究数据集"""
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 JSON 研究数据集", "emotion_research_data.json",
            "JSON 文件 (*.json);;所有文件 (*)"
        )
        if path:
            success = self.history_manager.export_research_dataset(path)
            if success:
                QMessageBox.information(self, "导出成功",
                    f"完整研究数据集已导出到:\n{path}\n\n包含原始模型输出等全部字段")
            else:
                QMessageBox.warning(self, "导出失败", "没有可导出的记录或导出过程出错")

    @exception_safe()
    def refresh_history(self):
        try:
            records = self.history_manager.get_records()
            self.history_list.clear()

            self.record_count_label.setText(f"共 {len(records)} 条记录")

            if not records:
                self.history_list.addItem("暂无历史记录")
                self.history_ids = []
            else:
                self.history_ids = []
                for record in records:
                    try:
                        timestamp = str(record.get('timestamp', ''))
                        emotion = str(record.get('主要情绪', record.get('main_emotion', '未知')))
                        score = float(record.get('情绪稳定度分数', record.get('anxiety_score', 0.0)))
                        level = str(record.get('情绪状态等级', record.get('emotion_level', '未知')))
                        record_id = str(record.get('id', ''))
                        self.history_ids.append(record_id)
                        item_text = f"{timestamp}  |  情绪: {emotion}  |  稳定度: {score:.1f}/10  |  状态: {level}"
                        self.history_list.addItem(item_text)
                    except Exception as e:
                        logger.warning(f"跳过无效记录: {str(e)}")
                        continue

            if MATPLOTLIB_AVAILABLE and hasattr(self, 'canvas'):
                self.canvas.plot_data(records)
        except Exception as e:
            logger.error(f"刷新历史记录异常: {str(e)}", exc_info=True)
            self.append_log(f"刷新历史记录失败: {str(e)}")

    def _on_history_cleanup_needed(self, count):
        """历史记录达到阈值时，通知用户即将自动清理"""
        try:
            QMessageBox.information(
                self, "历史记录管理",
                f"历史记录已达到50条，系统将自动删除最早的 {count} 条记录。\n\n"
                f"被删除记录的音频文件也会一并清除。"
            )
        except Exception:
            pass

    @exception_safe()
    def clear_all_history(self):
        try:
            reply = QMessageBox.question(
                self, "确认清空",
                "确定要清空所有历史记录和录音文件吗？\n\n"
                "此操作不可恢复！\n"
                "（AI模型文件不会被删除）",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.history_manager.delete_all_records()
                self.refresh_history()
                self.append_log("已清空所有历史记录和录音文件")
                logger.info("已清空所有历史记录")
                QMessageBox.information(self, "完成", "已清空所有历史记录和录音文件")
        except Exception as e:
            logger.error(f"清空记录异常: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "错误", f"清空记录失败: {str(e)}")

    @exception_safe()
    def get_history_data(self):
        return self.history_manager.get_records()

    @exception_safe()
    def _cleanup_temp_file(self, file_path):
        if file_path and isinstance(file_path, str):
            try:
                if is_safe_path(file_path) and os.path.exists(file_path):
                    safe_remove_file(file_path)
                    if file_path in self.temp_files:
                        self.temp_files.remove(file_path)
                    logger.info(f"清理临时文件: {file_path}")
            except Exception as e:
                logger.warning(f"清理临时文件失败: {file_path}, {str(e)}")

    @exception_safe()
    def cleanup_all_temp_files(self):
        for file_path in list(self.temp_files):
            self._cleanup_temp_file(file_path)

        temp_dir = get_temp_dir()
        if os.path.exists(temp_dir):
            try:
                for filename in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, filename)
                    try:
                        if is_safe_path(file_path) and os.path.isfile(file_path):
                            safe_remove_file(file_path)
                            logger.info(f"清理临时录音文件: {file_path}")
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"清理临时录音目录失败: {str(e)}")

    @exception_safe()
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_bg_widget'):
            self._bg_widget.setGeometry(self.centralWidget().rect())
        if hasattr(self, '_diag_stripe'):
            self._diag_stripe.setGeometry(self.centralWidget().rect())

    @exception_safe()
    def closeEvent(self, event):
        try:
            self._closing = True
            logger.info("正在关闭系统...")
            self.append_log("正在关闭系统...")

            if self.is_recording:
                if self.recording_thread and self.recording_thread.isRunning():
                    self.recording_thread.stop()
                    self.recording_thread.wait(2000)
                try:
                    self.recorder.stop_recording()
                except Exception:
                    pass

            if self.analysis_thread and self.analysis_thread.isRunning():
                self.analysis_thread.wait(2000)

            self.cleanup_all_temp_files()

            logger.info("系统已正常关闭")
            event.accept()
        except Exception as e:
            logger.error(f"关闭系统时异常: {str(e)}", exc_info=True)
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    app.setStyle("Fusion")
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
