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
7. 自定义控件：分数卡片（ScoreCard）、趋势图（MplCanvas）

界面设计特点：
- 采用卡片式布局，现代化渐变设计
- 左右分栏设计，左侧为主操作区，右侧为日志和指南
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
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QObject, QUrl
from PyQt5.QtGui import QFont, QPalette, QColor, QDesktopServices

from recorder import AudioRecorder
from emotion_recognizer import EmotionRecognizer
from history_manager import HistoryManager
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


class RecordingThread(QThread):
    """
    录音线程类

    在独立线程中执行音频录制操作，避免阻塞 GUI 界面。
    通过 Qt 信号与主线程通信：
    - duration_updated: 录音时长更新（每 100ms 触发一次）
    - recording_finished: 录音完成，返回音频文件路径
    - recording_error: 录音出错，返回错误信息

    参数：
        recorder (AudioRecorder): 音频录制器实例
        output_path (str): 录音保存路径
    """
    duration_updated = pyqtSignal(float)
    recording_finished = pyqtSignal(str)
    recording_error = pyqtSignal(str)

    def __init__(self, recorder, output_path):
        super().__init__()
        self.recorder = recorder
        self.output_path = output_path
        self._stop_requested = False

    def run(self):
        try:
            success = self.recorder.start_recording(self.output_path)
            if not success:
                self.recording_error.emit("录音启动失败，请检查麦克风是否正常连接")
                return

            while self.recorder.is_recording() and not self._stop_requested:
                duration = self.recorder.get_duration()
                self.duration_updated.emit(duration)
                self.msleep(100)

            output = self.recorder.stop_recording()
            if output:
                self.recording_finished.emit(output)
            else:
                self.recording_error.emit("录音保存失败，请检查磁盘空间")
        except Exception as e:
            logger.error(f"录音线程异常: {str(e)}", exc_info=True)
            self.recording_error.emit(f"录音异常: {str(e)}")

    def stop(self):
        self._stop_requested = True


class AnalysisThread(QThread):
    """
    情绪分析线程类

    在独立线程中执行 AI 模型推理，避免阻塞 GUI 界面。
    通过 Qt 信号与主线程通信：
    - analysis_finished: 分析完成，返回结果字典
    - analysis_error: 分析出错，返回错误信息

    参数：
        recognizer (EmotionRecognizer): 情绪识别器实例
        audio_path (str): 待分析的音频文件路径
    """
    analysis_finished = pyqtSignal(dict)
    analysis_error = pyqtSignal(str)

    def __init__(self, recognizer, audio_path):
        super().__init__()
        self.recognizer = recognizer
        self.audio_path = audio_path

    def run(self):
        try:
            if not os.path.exists(self.audio_path):
                self.analysis_error.emit("音频文件不存在，可能已被删除")
                return
            result = self.recognizer.predict(self.audio_path)
            self.analysis_finished.emit(result)
        except Exception as e:
            logger.error(f"分析线程异常: {str(e)}", exc_info=True)
            self.analysis_error.emit(f"分析异常: {str(e)}")


class ModelLoadThread(QThread):
    """
    模型加载线程类

    在独立线程中加载 AI 模型，避免阻塞 GUI 启动界面。
    通过 Qt 信号与主线程通信：
    - model_loaded: 模型加载完成（成功或失败）
    - progress_update: 加载进度更新

    参数：
        recognizer (EmotionRecognizer): 情绪识别器实例
    """
    model_loaded = pyqtSignal(bool, str)
    progress_update = pyqtSignal(str)

    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer

    def run(self):
        def on_progress(msg):
            self.progress_update.emit(msg)

        def on_loaded(success, error):
            self.model_loaded.emit(success, error or "")

        self.recognizer.add_progress_callback(on_progress)
        self.recognizer.load_model(callback=on_loaded)

        while self.recognizer.loading:
            self.msleep(100)


class MplCanvas(FigureCanvas):
    """
    matplotlib 图表画布类

    封装 matplotlib 的 FigureCanvas，用于在 PyQt5 界面中显示图表。
    主要功能：
    - 绘制情绪稳定度变化趋势图
    - 自动配置中文字体（微软雅黑/黑体）
    - 支持分区域着色（不同稳定度等级不同颜色）
    - 自动根据记录数调整 x 轴刻度

    参数：
        parent: 父窗口部件
        width: 图表宽度（英寸）
        height: 图表高度（英寸）
        dpi: 图表分辨率
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
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

    def plot_data(self, records):
        if not MATPLOTLIB_AVAILABLE:
            return

        try:
            self.axes.clear()

            if not records:
                self.axes.set_title("情绪稳定度变化趋势", fontsize=12, fontweight='bold', pad=15)
                self.axes.set_xlabel("记录序号", fontsize=10, labelpad=10)
                self.axes.set_ylabel("情绪稳定度", fontsize=10, labelpad=10)
                self.axes.set_ylim(0, 10)
                self.axes.grid(True, alpha=0.3, linestyle='--')
                self.axes.axhspan(0, 2, alpha=0.15, color='#27ae60')
                self.axes.axhspan(2, 4, alpha=0.15, color='#2ecc71')
                self.axes.axhspan(4, 6, alpha=0.15, color='#f1c40f')
                self.axes.axhspan(6, 8, alpha=0.15, color='#e74c3c')
                self.axes.axhspan(8, 10, alpha=0.15, color='#8b0000')
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#27ae60', alpha=0.5, label='非常稳定'),
                    Patch(facecolor='#2ecc71', alpha=0.5, label='稳定'),
                    Patch(facecolor='#f1c40f', alpha=0.5, label='轻微波动'),
                    Patch(facecolor='#e74c3c', alpha=0.5, label='不稳定'),
                    Patch(facecolor='#8b0000', alpha=0.5, label='情绪激烈')
                ]
                self.axes.legend(handles=legend_elements, loc='upper right', fontsize=8)
                self.fig.tight_layout()
                self.draw()
                return

            records_chronological = sorted(records, key=lambda x: x.get("timestamp", ""))
            n = len(records_chronological)
            x = list(range(1, n + 1))
            y = []
            for r in records_chronological:
                try:
                    score = r.get('情绪稳定度分数', r.get('anxiety_score', 0.0))
                    y.append(float(score))
                except (TypeError, ValueError):
                    y.append(0.0)

            self.axes.axhspan(0, 2, alpha=0.15, color='#27ae60')
            self.axes.axhspan(2, 4, alpha=0.15, color='#2ecc71')
            self.axes.axhspan(4, 6, alpha=0.15, color='#f1c40f')
            self.axes.axhspan(6, 8, alpha=0.15, color='#e74c3c')
            self.axes.axhspan(8, 10, alpha=0.15, color='#8b0000')

            self.axes.plot(x, y, color='#667eea', linewidth=2.5, marker='o',
                           markersize=7, markerfacecolor='#764ba2', markeredgecolor='white',
                           markeredgewidth=2)

            for i, (xi, yi) in enumerate(zip(x, y)):
                self.axes.annotate(f'{yi:.1f}', (xi, yi), textcoords="offset points",
                                  xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

            self.axes.set_title("情绪稳定度变化趋势", fontsize=13, fontweight='bold', pad=15)
            self.axes.set_xlabel("记录序号（按时间先后）", fontsize=10, labelpad=10)
            self.axes.set_ylabel("情绪稳定度 (0-10)", fontsize=10, labelpad=10)
            self.axes.set_ylim(0, 10.5)
            self.axes.set_xlim(0.5, n + 0.5)
            self.axes.set_xticks(x)
            if n > 20:
                self.axes.set_xticks(x[::max(1, n // 10)])
            self.axes.grid(True, alpha=0.3, linestyle='--')

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#27ae60', alpha=0.5, label='非常稳定 (0-2)'),
                Patch(facecolor='#2ecc71', alpha=0.5, label='稳定 (2-4)'),
                Patch(facecolor='#f1c40f', alpha=0.5, label='轻微波动 (4-6)'),
                Patch(facecolor='#e74c3c', alpha=0.5, label='不稳定 (6-8)'),
                Patch(facecolor='#8b0000', alpha=0.5, label='情绪激烈 (8-10)')
            ]
            self.axes.legend(handles=legend_elements, loc='upper right', fontsize=8)

            self.fig.tight_layout()
            self.draw()
        except Exception as e:
            logger.error(f"绘图失败: {str(e)}", exc_info=True)


class ScoreCard(QFrame):
    """
    分数卡片控件类

    自定义的分数展示卡片控件，用于在结果区域显示：
    - 情绪稳定度分数
    - 情绪状态等级
    - 主要情绪

    设计特点：
    - 圆角卡片样式，现代化设计
    - 大字号数字展示，醒目清晰
    - 支持自定义颜色和副标题
    - 支持重置为空状态

    参数：
        title (str): 卡片标题
        parent: 父窗口部件
    """
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("scoreCard")
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 22, 25, 22)
        layout.setSpacing(10)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("cardTitle")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Microsoft YaHei", 11))
        self.title_label.setMinimumHeight(25)
        layout.addWidget(self.title_label)

        self.value_label = QLabel("--")
        self.value_label.setObjectName("cardValue")
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setFont(QFont("Microsoft YaHei", 36, QFont.Bold))
        self.value_label.setMinimumHeight(60)
        self.value_label.setWordWrap(False)
        layout.addWidget(self.value_label, 1)

        self.sub_label = QLabel("")
        self.sub_label.setObjectName("cardSub")
        self.sub_label.setAlignment(Qt.AlignCenter)
        self.sub_label.setFont(QFont("Microsoft YaHei", 11))
        self.sub_label.setMinimumHeight(25)
        layout.addWidget(self.sub_label)

    def set_value(self, value, color=None, sub_text=""):
        self.value_label.setText(str(value))
        if color:
            self.value_label.setStyleSheet(f"color: {color};")
        if sub_text:
            self.sub_label.setText(sub_text)
            self.sub_label.setVisible(True)
        else:
            self.sub_label.setVisible(False)

    def reset(self):
        self.value_label.setText("--")
        self.value_label.setStyleSheet("")
        self.sub_label.setText("")
        self.sub_label.setVisible(False)


class WelcomeDialog(QDialog):
    """
    欢迎向导对话框类

    在首次运行程序时显示，向用户介绍：
    - 系统功能概述
    - AI 模型信息（emotion2vec_plus_large）
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

        title_label = QLabel("🎙️ 欢迎使用语音情绪识别系统")
        title_label.setFont(QFont("Microsoft YaHei", 22, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #667eea; margin-bottom: 10px;")
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
            model_group = QGroupBox("📦 选择AI模型")
            model_group.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
            model_layout = QVBoxLayout(model_group)
            model_layout.setContentsMargins(20, 25, 20, 20)
            model_text = QLabel(
                "<div style='line-height: 1.7; font-size: 11pt;'>"
                "<p><b>请选择要使用的 emotion2vec+ 模型：</b></p>"
                "<p>• 来源：ModelScope 达摩院（阿里巴巴）</p>"
                "<p>• 许可协议：Apache License 2.0（可免费使用）</p>"
                "<p style='color: #e67e22;'><b>⚠️ 重要提示：</b></p>"
                "<p>首次使用需要下载模型，请确保网络连接正常。模型下载完成后，"
                "以后启动程序就会很快啦！模型会保存在程序文件夨中，不会丢失。</p>"
                "</div>"
            )
            model_text.setWordWrap(True)
            model_text.setTextFormat(Qt.RichText)
            model_layout.addWidget(model_text)
        
            # 模型选择 ComboBox
            model_select_label = QLabel("🤖 选择模型：")
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
                    border: 2px solid #667eea;
                    border-radius: 8px;
                    padding: 8px 15px;
                    background-color: white;
                }
                QComboBox:hover {
                    border-color: #764ba2;
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

        tech_group = QGroupBox("🛠️ 技术栈")
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

        usage_group = QGroupBox("📖 使用方法（超简单！）")
        usage_group.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        usage_layout = QVBoxLayout(usage_group)
        usage_layout.setContentsMargins(20, 25, 20, 20)
        usage_text = QLabel(
            "<div style='line-height: 2.0; font-size: 12pt;'>"
            "1️⃣ 等待模型加载完成（首次需下载，会显示进度）<br>"
            "2️⃣ 点击绿色的「开始录音」按钮<br>"
            "3️⃣ 自然地说出您现在的感受（建议 3-30 秒）<br>"
            "4️⃣ 点击红色的「停止录音」按钮<br>"
            "5️⃣ 稍等几秒，就能看到分析结果啦！<br><br>"
            "<b>💡 小贴士：</b>在安静的环境下录音，效果会更好哦～"
            "</div>"
        )
        usage_text.setWordWrap(True)
        usage_text.setTextFormat(Qt.RichText)
        usage_layout.addWidget(usage_text)
        content_layout.addWidget(usage_group)

        tips_group = QGroupBox("📂 数据管理")
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

        disclaimer_group = QGroupBox("⚠️ 重要声明")
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
            start_btn = QPushButton("🚀 开始使用（下载模型）")
            start_btn.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
            start_btn.setMinimumHeight(50)
            start_btn.setMinimumWidth(250)
            start_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #667eea, stop:1 #764ba2);
                    color: white;
                    border: none;
                    border-radius: 12px;
                    padding: 12px 30px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #764ba2, stop:1 #667eea);
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
                    background-color: #667eea;
                    color: white;
                    border: none;
                    border-radius: 10px;
                    padding: 10px 30px;
                }
                QPushButton:hover {
                    background-color: #764ba2;
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
        self.setWindowTitle("🤖 切换AI模型")
        self.setMinimumSize(550, 420)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(25, 25, 25, 25)

        title_label = QLabel("🤖 选择情绪识别模型")
        title_label.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #667eea; margin-bottom: 5px;")
        layout.addWidget(title_label)

        desc_label = QLabel("选择不同的模型会影响识别精度和运行速度，切换后需要重新加载模型。")
        desc_label.setFont(QFont("Microsoft YaHei", 10))
        desc_label.setStyleSheet("color: #6c757d;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        from emotion_recognizer import AVAILABLE_MODELS

        self.btn_group = QButtonGroup(self)
        models_info = [
            ("emotion2vec_plus_large", "🌟 Large（大型模型）", "~1GB", "精度最高，推荐使用"),
            ("emotion2vec_plus_base", "⭐ Base（基础模型）", "~500MB", "速度与精度均衡"),
            ("emotion2vec_plus_seed", "⚡ Seed（最小模型）", "~200MB", "速度最快，适合低配置设备"),
        ]

        for model_name, display, size, desc in models_info:
            downloaded = is_model_downloaded(model_name)
            is_current = (model_name == self.current_model)

            radio = QRadioButton()
            radio.setProperty("model_name", model_name)
            if is_current:
                radio.setChecked(True)

            # 构建卡片展示文本
            status_text = "✅ 当前使用" if is_current else ("✔ 已下载" if downloaded else "⚠ 需要下载")
            status_color = "#27ae60" if (is_current or downloaded) else "#e67e22"
            radio.setText(f"{display}  |  大小: {size}  |  {desc}")
            radio.setFont(QFont("Microsoft YaHei", 11))
            radio.setMinimumHeight(50)
            radio.setStyleSheet(f"""
                QRadioButton {{
                    padding: 12px 15px;
                    border: 2px solid {'#667eea' if is_current else '#e9ecef'};
                    border-radius: 10px;
                    background-color: {'#f0f3ff' if is_current else 'white'};
                }}
                QRadioButton:hover {{
                    border-color: #667eea;
                    background-color: #f8f9ff;
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
                background-color: #e9ecef;
                color: #495057;
                border: none;
                border-radius: 8px;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #dee2e6;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self.confirm_btn = QPushButton("✅ 确认切换")
        self.confirm_btn.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.confirm_btn.setMinimumHeight(40)
        self.confirm_btn.setMinimumWidth(140)
        self.confirm_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #764ba2, stop:1 #667eea);
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

        info_group = QGroupBox("💾 数据存储位置（便携模式）")
        info_group.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        info_layout = QVBoxLayout(info_group)

        app_dir = get_app_dir()
        data_dir = get_user_data_dir()

        def path_to_url(p):
            return QUrl.fromLocalFile(os.path.abspath(p)).toString()

        path_info = QLabel(
            f"<div style='line-height: 1.8; font-size: 11pt;'>"
            f"<p><b>程序目录：</b><br>"
            f"<a href='{path_to_url(app_dir)}' style='color: #667eea;'>{app_dir}</a></p>"
            f"<p><b>数据目录（所有用户数据保存在此）：</b><br>"
            f"<a href='{path_to_url(data_dir)}' style='color: #667eea;'>{data_dir}</a></p>"
            f"<p><b>数据结构说明：</b></p>"
            f"<ul style='margin: 5px 0; padding-left: 20px; line-height: 2.0;'>"
            f"<li>📁 <b>recordings/</b> —— 您的录音文件（可删除）</li>"
            f"<li>📁 <b>models/</b> —— AI模型文件（<span style='color: #e74c3c; font-weight: bold;'>不可删除，删除后需重新下载约1GB</span>）</li>"
            f"<li>📁 <b>logs/</b> —— 程序运行日志（可清除）</li>"
            f"<li>📁 <b>temp/</b> —— 临时文件（可清除）</li>"
            f"<li>📁 <b>cache/</b> —— 缓存文件（可清除）</li>"
            f"<li>📄 <b>history.json</b> —— 历史记录（可删除）</li>"
            f"</ul>"
            f"</div>"
        )
        path_info.setTextFormat(Qt.RichText)
        path_info.setTextInteractionFlags(Qt.TextBrowserInteraction)
        path_info.setOpenExternalLinks(True)
        path_info.setWordWrap(True)
        info_layout.addWidget(path_info)

        layout.addWidget(info_group)

        self.stats_label = QLabel("")
        self.stats_label.setFont(QFont("Microsoft YaHei", 11))
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("padding: 10px; background-color: #f8f9fa; border-radius: 8px;")
        layout.addWidget(self.stats_label)

        actions_group = QGroupBox("🗑️ 数据清理操作")
        actions_group.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        actions_layout = QVBoxLayout(actions_group)
        actions_layout.setSpacing(10)
        actions_layout.setContentsMargins(15, 25, 15, 15)

        del_all_btn = QPushButton("🗑️ 清空所有历史记录和录音")
        del_all_btn.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        del_all_btn.setMinimumHeight(45)
        del_all_btn.setObjectName("dangerBtn")
        del_all_btn.setCursor(Qt.PointingHandCursor)
        del_all_btn.clicked.connect(self.delete_all_records)
        actions_layout.addWidget(del_all_btn)

        clear_all_btn = QPushButton("🧹 一键清理系统垃圾")
        clear_all_btn.setFont(QFont("Microsoft YaHei", 11))
        clear_all_btn.setMinimumHeight(45)
        clear_all_btn.setCursor(Qt.PointingHandCursor)
        clear_all_btn.setToolTip("一键清除临时文件、缓存文件和日志文件")
        clear_all_btn.clicked.connect(self.clear_all_junk)
        actions_layout.addWidget(clear_all_btn)

        open_folder_btn = QPushButton("📂 打开数据文件夹")
        open_folder_btn.setFont(QFont("Microsoft YaHei", 11))
        open_folder_btn.setMinimumHeight(45)
        open_folder_btn.setCursor(Qt.PointingHandCursor)
        open_folder_btn.clicked.connect(self.open_data_folder)
        actions_layout.addWidget(open_folder_btn)

        layout.addWidget(actions_group)
        layout.addStretch()

        scroll_area.setWidget(scroll_content)
        outer_layout.addWidget(scroll_area, 1)

        refresh_btn = QPushButton("🔄 刷新统计")
        refresh_btn.setFont(QFont("Microsoft YaHei", 10))
        refresh_btn.setMinimumHeight(40)
        refresh_btn.setMinimumWidth(120)
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.clicked.connect(self.refresh_stats)

        close_btn = QPushButton("关闭")
        close_btn.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        close_btn.setMinimumHeight(40)
        close_btn.setMinimumWidth(120)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.clicked.connect(self.accept)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(refresh_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        outer_layout.addLayout(btn_layout)

    def refresh_stats(self):
        try:
            stats = get_storage_stats()
            stats_text = "<b>📊 存储统计：</b><br>"
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
            self, "⚠️ 确认删除",
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


class MainWindow(QMainWindow):
    """
    主窗口类

    应用程序的主界面窗口，集成所有功能模块：
    - 菜单栏：文件菜单（数据管理、退出）、帮助菜单
    - 实时检测页面：录音控制、情绪分析、结果展示、运行日志
    - 历史报告页面：历史记录列表、情绪趋势图
    - 底部操作区：数据管理按钮、版权声明

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
        self.history_manager = HistoryManager()
        self.temp_files = []
        self._closing = False

        self.log_signal.connect(self._append_log_safe)
        self.model_loaded_signal.connect(self._on_model_loaded_safe)
        self.model_progress_signal.connect(self._append_log_safe)

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
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_text.append(f"[{timestamp}] {msg}")
            self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        except Exception:
            pass

    def append_log(self, msg):
        self.log_signal.emit(msg)

    def _on_model_loaded_safe(self, success, error):
        if self._closing:
            return
        # 重新启用模型切换按钮
        self.model_switch_btn.setEnabled(True)
        if success:
            self._append_log_safe("✅ 情绪识别模型加载完成，准备就绪！")
            self.status_label.setText("● 准备就绪")
            self.status_label.setStyleSheet("color: #a8ffb8; background-color: rgba(255,255,255,0.15); padding: 8px 20px; border-radius: 20px;")
            self.record_btn.setText("🎤  点击开始录音")
            self.record_btn.setProperty("isRecording", "false")
            self.record_btn.style().unpolish(self.record_btn)
            self.record_btn.style().polish(self.record_btn)
            self.record_btn.setEnabled(True)
            logger.info("情绪识别模型加载成功")
        else:
            error_msg = f"模型加载失败: {error}"
            self._append_log_safe(f"❌ {error_msg}")
            logger.error(error_msg)
            self.status_label.setText("● 模型加载失败")
            self.status_label.setStyleSheet("color: #ff6b6b; background-color: rgba(255,255,255,0.15); padding: 8px 20px; border-radius: 20px;")
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
        self.setWindowTitle("语音情绪识别系统 v1.0 - 便携版")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")
        header_frame.setMinimumHeight(70)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(25, 12, 25, 12)

        title_label = QLabel("🎙️ 语音情绪识别系统")
        title_label.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        title_label.setObjectName("headerTitle")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        # 模型切换按钮
        self.model_switch_btn = QPushButton("🤖 切换模型")
        self.model_switch_btn.setFont(QFont("Microsoft YaHei", 10))
        self.model_switch_btn.setCursor(Qt.PointingHandCursor)
        self.model_switch_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255,255,255,0.2);
                color: white;
                border: 1px solid rgba(255,255,255,0.4);
                border-radius: 15px;
                padding: 6px 16px;
            }
            QPushButton:hover {
                background-color: rgba(255,255,255,0.3);
                border-color: rgba(255,255,255,0.6);
            }
        """)
        self.model_switch_btn.clicked.connect(self.show_model_switch_dialog)
        header_layout.addWidget(self.model_switch_btn)

        self.status_label = QLabel("● 正在启动...")
        self.status_label.setFont(QFont("Microsoft YaHei", 12))
        self.status_label.setObjectName("statusLoading")
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

        data_mgr_action = QAction("💾 数据管理", self)
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
        self.status_label.setStyleSheet("color: #ffd700; background-color: rgba(255,255,255,0.15); padding: 8px 20px; border-radius: 20px;")

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

        control_group = QGroupBox("🎤 录音控制")
        control_group.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        control_group.setObjectName("controlGroup")
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(12)
        control_layout.setContentsMargins(20, 25, 20, 20)

        self.record_btn = QPushButton("🎤  模型加载中...")
        self.record_btn.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        self.record_btn.setMinimumHeight(100)
        self.record_btn.setObjectName("recordBtn")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        self.record_btn.setCursor(Qt.PointingHandCursor)
        control_layout.addWidget(self.record_btn)

        info_grid = QGridLayout()
        info_grid.setSpacing(12)
        info_grid.setColumnStretch(1, 1)

        duration_title = QLabel("⏱️ 录音时长")
        duration_title.setFont(QFont("Microsoft YaHei", 11))
        info_grid.addWidget(duration_title, 0, 0)

        self.duration_display = QLabel("00:00")
        self.duration_display.setFont(QFont("Consolas", 22, QFont.Bold))
        self.duration_display.setMinimumWidth(120)
        self.duration_display.setAlignment(Qt.AlignCenter)
        self.duration_display.setObjectName("durationLabel")
        info_grid.addWidget(self.duration_display, 0, 1)

        progress_title = QLabel("📊 处理进度")
        progress_title.setFont(QFont("Microsoft YaHei", 11))
        info_grid.addWidget(progress_title, 1, 0)

        self.record_progress = QProgressBar()
        self.record_progress.setRange(0, 100)
        self.record_progress.setValue(0)
        self.record_progress.setObjectName("recordProgress")
        self.record_progress.setTextVisible(False)
        self.record_progress.setMinimumHeight(24)
        info_grid.addWidget(self.record_progress, 1, 1)

        control_layout.addLayout(info_grid)
        left_layout.addWidget(control_group)

        result_group = QGroupBox("📊 检测结果")
        result_group.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        result_group.setObjectName("resultGroup")
        self.result_group = result_group
        result_layout = QVBoxLayout(result_group)
        result_layout.setSpacing(15)
        result_layout.setContentsMargins(20, 25, 20, 20)

        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(12)

        self.score_card = ScoreCard("情绪稳定度 (0-10)")
        self.level_card = ScoreCard("情绪状态")
        self.emotion_card = ScoreCard("主要情绪")
        self.compound_card = ScoreCard("复合情绪")

        cards_layout.addWidget(self.score_card)
        cards_layout.addWidget(self.level_card)
        cards_layout.addWidget(self.emotion_card)
        cards_layout.addWidget(self.compound_card)

        result_layout.addLayout(cards_layout)

        self.warning_label = QLabel("")
        self.warning_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.warning_label.setAlignment(Qt.AlignCenter)
        self.warning_label.setObjectName("warningLabel")
        self.warning_label.setMinimumHeight(50)
        self.warning_label.setWordWrap(True)
        self.warning_label.hide()
        result_layout.addWidget(self.warning_label)

        prob_title = QLabel("📈 情绪概率分布")
        prob_title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        result_layout.addWidget(prob_title)

        self.prob_bars = {}
        self.prob_labels = {}
        emotions = [("平静", "#95a5a6"), ("开心", "#f1c40f"), ("惊讶", "#1abc9c"),
                    ("悲伤", "#3498db"), ("愤怒", "#e74c3c"), ("恐惧", "#9b59b6"),
                    ("厌恶", "#e67e22")]
        for emotion, color in emotions:
            bar_row = QHBoxLayout()
            bar_row.setSpacing(10)

            label = QLabel(emotion)
            label.setFont(QFont("Microsoft YaHei", 11))
            label.setMinimumWidth(50)
            label.setMaximumWidth(50)
            bar_row.addWidget(label)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setObjectName(f"probBar_{emotion}")
            bar.setMinimumHeight(28)
            bar.setMaximumHeight(28)
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

        suggestion_group = QGroupBox("💡 调节建议")
        suggestion_group.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        suggestion_group.setObjectName("suggestionGroup")
        suggestion_layout = QVBoxLayout(suggestion_group)
        suggestion_layout.setContentsMargins(15, 20, 15, 15)
        self.suggestion_text = QTextEdit()
        self.suggestion_text.setReadOnly(True)
        self.suggestion_text.setMinimumHeight(120)
        self.suggestion_text.setPlaceholderText("等待模型加载完成后，进行录音检测即可看到结果...")
        self.suggestion_text.setFont(QFont("Microsoft YaHei", 11))
        self.suggestion_text.setObjectName("suggestionText")
        suggestion_layout.addWidget(self.suggestion_text)
        result_layout.addWidget(suggestion_group)

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

        log_group = QGroupBox("📝 运行日志")
        log_group.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        log_group.setObjectName("logGroup")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(15, 25, 15, 15)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setObjectName("logText")
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group, 1)

        quick_guide = QGroupBox("📖 快速指南")
        quick_guide.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        quick_guide.setObjectName("guideGroup")
        guide_layout = QVBoxLayout(quick_guide)
        guide_layout.setContentsMargins(15, 25, 15, 15)
        guide_text = QLabel(
            "<div style='line-height: 1.8; font-size: 11pt;'>"
            "<b>使用流程：</b><br>"
            "1️⃣ 等待模型加载完成（首次需下载）<br>"
            "2️⃣ 点击绿色「开始录音」按钮<br>"
            "3️⃣ 自然地说出您的感受<br>"
            "4️⃣ 点击红色「停止录音」按钮<br>"
            "5️⃣ 等待几秒查看分析结果<br><br>"
            "<b>提示：</b>录音时长建议3-30秒，保持环境安静<br>"
            "<b>便携模式：</b>所有数据保存在程序文件夹"
            "</div>"
        )
        guide_text.setWordWrap(True)
        guide_text.setTextFormat(Qt.RichText)
        guide_layout.addWidget(guide_text)
        right_layout.addWidget(quick_guide)

        content_splitter.addWidget(right_panel)

        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 1)
        content_splitter.setSizes([850, 350])

        layout.addWidget(content_splitter)
        self.tab_widget.addTab(realtime_widget, "🎙️ 实时检测")

    @exception_safe()
    def create_history_tab(self):
        history_widget = QWidget()
        layout = QVBoxLayout(history_widget)
        layout.setSpacing(12)

        history_btn_layout = QHBoxLayout()
        history_btn_layout.setSpacing(10)

        refresh_btn = QPushButton("🔄 刷新列表")
        refresh_btn.setFont(QFont("Microsoft YaHei", 11))
        refresh_btn.setMinimumHeight(38)
        refresh_btn.setMinimumWidth(120)
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.clicked.connect(self.refresh_history)
        history_btn_layout.addWidget(refresh_btn)

        history_btn_layout.addStretch()

        record_count_label = QLabel("")
        record_count_label.setFont(QFont("Microsoft YaHei", 10))
        record_count_label.setObjectName("countLabel")
        self.record_count_label = record_count_label
        history_btn_layout.addWidget(record_count_label)

        layout.addLayout(history_btn_layout)

        splitter = QSplitter(Qt.Vertical)
        splitter.setObjectName("historySplitter")

        list_group = QGroupBox("📋 历史记录")
        list_group.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        list_group.setObjectName("historyGroup")
        list_layout = QVBoxLayout(list_group)
        list_layout.setContentsMargins(15, 25, 15, 15)
        self.history_list = QListWidget()
        self.history_list.setFont(QFont("Microsoft YaHei", 11))
        self.history_list.setObjectName("historyList")
        self.history_list.setMinimumHeight(220)
        self.history_list.setSpacing(2)
        self.history_list.setSelectionMode(QListWidget.NoSelection)
        list_layout.addWidget(self.history_list)
        splitter.addWidget(list_group)

        plot_group = QGroupBox("📈 情绪趋势图")
        plot_group.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        plot_group.setObjectName("plotGroup")
        plot_layout = QVBoxLayout(plot_group)
        plot_layout.setContentsMargins(15, 25, 15, 15)

        if MATPLOTLIB_AVAILABLE:
            self.canvas = MplCanvas(self, width=10, height=5, dpi=100)
            plot_layout.addWidget(self.canvas)
        else:
            plot_placeholder = QLabel(
                "📊 绘图库未安装\n\n趋势图功能暂不可用"
            )
            plot_placeholder.setAlignment(Qt.AlignCenter)
            plot_placeholder.setFont(QFont("Microsoft YaHei", 12))
            plot_placeholder.setStyleSheet("color: #7f8c8d; padding: 40px;")
            plot_layout.addWidget(plot_placeholder)

        splitter.addWidget(plot_group)
        splitter.setSizes([350, 450])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)
        self.tab_widget.addTab(history_widget, "📋 历史报告")

    @exception_safe()
    def create_bottom_buttons(self, parent_layout):
        bottom_container = QVBoxLayout()
        bottom_container.setSpacing(8)

        btn_frame = QFrame()
        btn_frame.setObjectName("bottomBtnFrame")
        btn_frame.setMinimumHeight(60)
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setSpacing(15)
        btn_layout.setContentsMargins(20, 8, 20, 8)

        data_btn = QPushButton("💾  数据管理")
        data_btn.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        data_btn.setMinimumHeight(48)
        data_btn.setMinimumWidth(150)
        data_btn.setObjectName("dataBtn")
        data_btn.clicked.connect(self.show_data_manager)
        data_btn.setCursor(Qt.PointingHandCursor)
        btn_layout.addWidget(data_btn)

        btn_layout.addStretch()

        bottom_container.addWidget(btn_frame)

        current_model = load_model_config()
        self.credit_label = QLabel(
            f"Powered by {current_model} (ModelScope/达摩院, Apache-2.0) | "
            "便携模式 - 数据保存在程序目录 | 仅供个人非商用参考使用"
        )
        self.credit_label.setAlignment(Qt.AlignCenter)
        self.credit_label.setFont(QFont("Microsoft YaHei", 8))
        self.credit_label.setObjectName("creditLabel")
        self.credit_label.setWordWrap(True)
        bottom_container.addWidget(self.credit_label)

        parent_layout.addLayout(bottom_container)

    @exception_safe()
    def setup_styles(self):
        qss = """
        QMainWindow {
            background-color: #f5f7fa;
        }

        #headerFrame {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #667eea, stop:0.5 #764ba2, stop:1 #f093fb);
            border-radius: 16px;
        }

        #headerTitle {
            color: white;
        }

        #statusReady {
            color: #a8ffb8;
            background-color: rgba(255,255,255,0.15);
            padding: 8px 20px;
            border-radius: 20px;
        }

        #statusLoading {
            color: #ffd700;
            background-color: rgba(255,255,255,0.15);
            padding: 8px 20px;
            border-radius: 20px;
        }

        QMenuBar {
            background-color: white;
            border-bottom: 1px solid #e8ecf1;
            font-family: "Microsoft YaHei";
            font-size: 11px;
        }

        QMenuBar::item {
            padding: 10px 18px;
            background-color: transparent;
        }

        QMenuBar::item:selected {
            background-color: #667eea;
            color: white;
            border-radius: 6px;
        }

        QMenu {
            background-color: white;
            border: 1px solid #e8ecf1;
            font-family: "Microsoft YaHei";
            font-size: 11px;
            border-radius: 8px;
            padding: 5px;
        }

        QMenu::item {
            padding: 10px 25px;
            border-radius: 6px;
        }

        QMenu::item:selected {
            background-color: #667eea;
            color: white;
        }

        QTabWidget::pane {
            border: none;
            border-radius: 16px;
            background-color: transparent;
        }

        QTabBar::tab {
            background-color: #e8ecf1;
            padding: 14px 35px;
            margin-right: 5px;
            border-top-left-radius: 12px;
            border-top-right-radius: 12px;
            font-weight: bold;
            font-size: 13px;
        }

        QTabBar::tab:selected {
            background-color: white;
            border-bottom: 4px solid #667eea;
            color: #667eea;
        }

        QTabBar::tab:hover:!selected {
            background-color: #d5dbe3;
        }

        QGroupBox {
            border: 2px solid #e8ecf1;
            border-radius: 16px;
            margin-top: 12px;
            padding-top: 20px;
            background-color: white;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 20px;
            padding: 0 15px;
            color: #2c3e50;
        }

        #recordBtn {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #95a5a6, stop:1 #7f8c8d);
            color: white;
            border: none;
            border-radius: 20px;
            padding: 20px;
        }

        #recordBtn:enabled {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #43e97b, stop:1 #38f9d7);
        }

        #recordBtn:enabled:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #3dd670, stop:1 #32e4c5);
        }

        #recordBtn:enabled:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #35c064, stop:1 #2ccdb0);
        }

        #recordBtn[isRecording="true"] {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #ff6b6b, stop:1 #ee5a6f);
        }

        #recordBtn[isRecording="true"]:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #ff5252, stop:1 #e0485c);
        }

        #durationLabel {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #f8f9fa, stop:1 #e9ecef);
            border-radius: 12px;
            padding: 10px 20px;
            color: #2c3e50;
            border: 2px solid #dee2e6;
        }

        #recordProgress {
            border: none;
            border-radius: 12px;
            background-color: #e9ecef;
        }

        #recordProgress::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #667eea, stop:1 #764ba2);
            border-radius: 12px;
        }

        #scoreCard {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #ffffff, stop:1 #f8f9fa);
            border-radius: 16px;
            border: 2px solid #e9ecef;
        }

        #cardTitle {
            color: #6c757d;
        }

        #cardValue {
            color: #667eea;
        }

        #cardSub {
            color: #adb5bd;
        }

        #warningLabel {
            border-radius: 12px;
            padding: 12px 20px;
        }

        QProgressBar {
            border: 2px solid #e9ecef;
            border-radius: 8px;
            background-color: #f8f9fa;
            text-align: center;
            font-weight: bold;
            font-size: 10pt;
        }

        QProgressBar::chunk {
            border-radius: 6px;
            margin: 2px;
        }

        #probBar_平静::chunk { background-color: #95a5a6; }
        #probBar_开心::chunk { background-color: #f1c40f; }
        #probBar_惊讶::chunk { background-color: #1abc9c; }
        #probBar_悲伤::chunk { background-color: #3498db; }
        #probBar_愤怒::chunk { background-color: #e74c3c; }
        #probBar_恐惧::chunk { background-color: #9b59b6; }
        #probBar_厌恶::chunk { background-color: #e67e22; }

        #probLabel_平静 { color: #95a5a6; }
        #probLabel_开心 { color: #d4ac0d; }
        #probLabel_惊讶 { color: #16a085; }
        #probLabel_悲伤 { color: #2980b9; }
        #probLabel_愤怒 { color: #c0392b; }
        #probLabel_恐惧 { color: #8e44ad; }
        #probLabel_厌恶 { color: #d35400; }

        #suggestionText {
            background-color: #f8f9fa;
            border: none;
            border-radius: 10px;
            padding: 12px;
            color: #495057;
            line-height: 1.6;
        }

        #logText {
            background-color: #2d3436;
            color: #00b894;
            border: none;
            border-radius: 12px;
            padding: 15px;
        }

        #historyList {
            border: 2px solid #e9ecef;
            border-radius: 12px;
            background-color: #fafbfc;
            padding: 8px;
        }

        #historyList::item {
            padding: 12px 15px;
            border-bottom: 1px solid #f1f3f5;
            border-radius: 8px;
            margin: 2px 0;
        }

        #historyList::item:selected {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #667eea, stop:1 #764ba2);
            color: white;
        }

        #historyList::item:hover:!selected {
            background-color: #f1f3f5;
        }

        #countLabel {
            color: #6c757d;
            padding: 5px 10px;
        }

        #dangerBtn {
            background-color: #e74c3c;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 8px 20px;
        }

        #dangerBtn:hover {
            background-color: #c0392b;
        }

        QScrollArea {
            border: none;
            background-color: transparent;
        }

        QScrollBar:vertical {
            border: none;
            background: #f1f3f5;
            width: 10px;
            border-radius: 5px;
        }

        QScrollBar::handle:vertical {
            background: #ced4da;
            border-radius: 5px;
            min-height: 30px;
        }

        QScrollBar::handle:vertical:hover {
            background: #adb5bd;
        }

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
        }

        #bottomBtnFrame {
            background-color: white;
            border-radius: 16px;
            border: 2px solid #e9ecef;
        }

        #dataBtn {
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 30px;
        }

        #dataBtn:hover {
            background-color: #2980b9;
        }

        #creditLabel {
            color: #868e96;
            padding: 5px;
        }

        QPushButton {
            outline: none;
        }

        QPushButton:disabled {
            background-color: #ced4da !important;
            color: #868e96 !important;
            cursor: not-allowed;
        }

        QSplitter::handle {
            background-color: transparent;
            width: 8px;
        }

        QSplitter::handle:hover {
            background-color: #dee2e6;
            border-radius: 4px;
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
        self.record_btn.setText("🎤  点击开始录音")
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
            self.record_btn.setText("⏹️  点击停止录音")
            self.record_btn.setProperty("isRecording", "true")
            self.record_btn.style().unpolish(self.record_btn)
            self.record_btn.style().polish(self.record_btn)
            self.record_progress.setRange(0, 0)
            self.duration_display.setText("00:00")
            self.status_label.setText("● 正在录音...")
            self.status_label.setStyleSheet("color: #ff6b6b; background-color: rgba(255,255,255,0.15); padding: 8px 20px; border-radius: 20px;")

            self.recording_thread.start()

            import time
            time.sleep(0.1)
            if not self.recording_thread.isRunning():
                raise Exception("录音线程启动失败")

            self.record_btn.setEnabled(True)
            self.append_log(f"✅ 录音已开始")
            logger.info(f"开始录音: {self.current_audio_path}")
        except Exception as e:
            logger.error(f"启动录音失败: {str(e)}", exc_info=True)
            self.append_log(f"❌ 启动录音失败: {str(e)}")
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
            self.record_btn.setText("💾  正在保存...")
            self.record_btn.setEnabled(False)
            self.status_label.setText("● 处理中...")
            self.status_label.setStyleSheet("color: #f39c12; background-color: rgba(255,255,255,0.15); padding: 8px 20px; border-radius: 20px;")
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
            self.append_log(f"❌ 停止录音异常: {str(e)}")
            QMessageBox.critical(self, "错误", f"停止录音失败:\n{str(e)}")
            self._reset_recording_ui()

    @exception_safe()
    def on_duration_updated(self, duration):
        minutes = int(duration) // 60
        seconds = int(duration) % 60
        self.duration_display.setText(f"{minutes:02d}:{seconds:02d}")

    @exception_safe()
    def on_recording_finished(self, output_path):
        self.is_recording = False
        self.record_btn.setText("🎤  点击开始录音")
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
                self.status_label.setStyleSheet("color: #a8ffb8; background-color: rgba(255,255,255,0.15); padding: 8px 20px; border-radius: 20px;")
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
        self.status_label.setStyleSheet("color: #e74c3c; background-color: rgba(255,255,255,0.15); padding: 8px 20px; border-radius: 20px;")
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

                self.score_card.set_value(f"{score:.1f}", color, "分")
                self.level_card.set_value(level, color)
                self.emotion_card.set_value(main_emotion, "#3498db", f"置信度 {confidence:.1%}")

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

                self.result_group.setStyleSheet(f"""
                    QGroupBox {{
                        border: 3px solid {border_color};
                        border-radius: 16px;
                        margin-top: 12px;
                        padding-top: 20px;
                        background-color: {bg_color};
                    }}
                    QGroupBox::title {{
                        subcontrol-origin: margin;
                        left: 20px;
                        padding: 0 15px;
                        color: #2c3e50;
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

                tips_text = f"<div style='font-size: 11pt; line-height: 1.8;'>"
                tips_text += f"<b>检测结果：</b>主要情绪为「{main_emotion}」，置信度 {confidence:.1%}"
                if mixed_emotions:
                    mixed_str = "、".join([f"{e}({p*100:.0f}%)" for e, p in mixed_emotions[:3]])
                    tips_text += f"，同时检测到混合情绪：{mixed_str}"

                # 复合情绪展示（变量已在上方卡片区域定义）
                if compound_emotion and compound_detail:
                    tips_text += f"<br><b>🧠 复合情绪：</b>「{compound_emotion}」——{compound_detail.get('desc', '')}"

                tips_text += f"<br><br><b>情绪稳定度：</b>{score:.1f}/10 分（分数越高表示情绪波动越大）<br><br>"
                tips_text += f"<b>💡 调节建议：</b><br>{advice}"

                # 获取分层建议（即时建议 + 长期建议）
                from relaxation_tips import get_tips
                tips_result = get_tips(
                    main_emotion, score,
                    compound_emotion=compound_emotion if compound_emotion else None,
                    mixed_emotions=mixed_emotions
                )

                if tips_result.get('immediate'):
                    tips_text += "<br><br><b>🎯 即时调节建议：</b><ul>"
                    for tip in tips_result['immediate']:
                        tips_text += f"<li>{tip}</li>"
                    tips_text += "</ul>"

                if tips_result.get('compound_advice'):
                    tips_text += f"<br><b>🔗 复合情绪解读：</b>{tips_result['compound_advice']}"

                if tips_result.get('long_term'):
                    tips_text += "<br><br><b>🌱 长期建议：</b><ul>"
                    for tip in tips_result['long_term']:
                        tips_text += f"<li>{tip}</li>"
                    tips_text += "</ul>"

                tips_text += "</div>"
                self.suggestion_text.setHtml(tips_text)

                result_copy = dict(result)
                result_copy['suggestion_text'] = tips_text
                result_copy['anxiety_score'] = score
                self.history_manager.add_record(result_copy)

                if score >= 6.5:
                    self.warning_label.show()
                    if score >= 8:
                        self.warning_label.setText("⚠️ 情绪波动较大！建议立即进行深呼吸放松，必要时寻求亲友陪伴或专业帮助")
                        self.warning_label.setStyleSheet("""
                            background-color: #f8d7da;
                            color: #721c24;
                            border: 2px solid #dc3545;
                            border-radius: 12px;
                            padding: 12px 20px;
                            font-weight: bold;
                        """)
                        self.status_label.setText("● 情绪不稳定")
                        self.status_label.setStyleSheet("color: #ff6b6b; background-color: rgba(255,255,255,0.15); padding: 8px 20px; border-radius: 20px;")
                    else:
                        self.warning_label.setText("⚠️ 情绪存在一定波动，建议适当休息放松")
                        self.warning_label.setStyleSheet("""
                            background-color: #fff3cd;
                            color: #856404;
                            border: 2px solid #ffc107;
                            border-radius: 12px;
                            padding: 12px 20px;
                            font-weight: bold;
                        """)
                        self.status_label.setText("● 情绪波动")
                        self.status_label.setStyleSheet("color: #f39c12; background-color: rgba(255,255,255,0.15); padding: 8px 20px; border-radius: 20px;")
                else:
                    self.warning_label.hide()
                    self.status_label.setText("● 情绪状态良好")
                    self.status_label.setStyleSheet("color: #a8ffb8; background-color: rgba(255,255,255,0.15); padding: 8px 20px; border-radius: 20px;")

                self.append_log(f"分析完成 - 稳定度分数: {score:.1f}/10, 情绪状态: {level}, 主要情绪: {main_emotion}" + (f", 复合情绪: {compound_emotion}" if compound_emotion else ""))
                logger.info(f"分析完成 - 稳定度: {score:.1f}, 状态: {level}, 情绪: {main_emotion}" + (f", 复合情绪: {compound_emotion}" if compound_emotion else ""))
            else:
                error_msg = result.get('error', '未知错误')
                QMessageBox.warning(self, "分析错误", f"情绪分析失败:\n{error_msg}")
                self.append_log(f"分析错误: {error_msg}")
                logger.error(f"分析失败: {error_msg}")
                self.status_label.setText("● 分析失败")
                self.status_label.setStyleSheet("color: #e74c3c; background-color: rgba(255,255,255,0.15); padding: 8px 20px; border-radius: 20px;")
        except Exception as e:
            logger.error(f"处理分析结果异常: {str(e)}", exc_info=True)
            self.append_log(f"处理分析结果异常: {str(e)}")
            QMessageBox.critical(self, "错误", f"处理分析结果失败:\n{str(e)}")
            self.status_label.setText("● 处理错误")
            self.status_label.setStyleSheet("color: #e74c3c; background-color: rgba(255,255,255,0.15); padding: 8px 20px; border-radius: 20px;")

    @exception_safe()
    def on_tab_changed(self, index):
        try:
            tab_text = self.tab_widget.tabText(index)
            if "历史报告" in tab_text:
                self.refresh_history()
        except Exception as e:
            logger.error(f"切换标签页异常: {str(e)}", exc_info=True)

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
                        item_text = f"⏰ {timestamp}  |  😊 情绪: {emotion}  |  📊 稳定度: {score:.1f}/10  |  🎯 状态: {level}"
                        self.history_list.addItem(item_text)
                    except Exception as e:
                        logger.warning(f"跳过无效记录: {str(e)}")
                        continue

            if MATPLOTLIB_AVAILABLE and hasattr(self, 'canvas'):
                self.canvas.plot_data(records)
        except Exception as e:
            logger.error(f"刷新历史记录异常: {str(e)}", exc_info=True)
            self.append_log(f"刷新历史记录失败: {str(e)}")

    @exception_safe()
    def clear_all_history(self):
        try:
            reply = QMessageBox.question(
                self, "⚠️ 确认清空",
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
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
