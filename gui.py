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
    QRadioButton, QButtonGroup, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QObject, QUrl, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtGui import QFont, QPalette, QColor, QDesktopServices, QPainter, QBrush, QPen, QPainterPath

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
        """绘制稳定度背景色带"""
        self.axes.axhspan(0, 2, alpha=0.15, color='#1B5E20', zorder=0)
        self.axes.axhspan(2, 4, alpha=0.15, color='#4CAF50', zorder=0)
        self.axes.axhspan(4, 6, alpha=0.15, color='#F4D03F', zorder=0)
        self.axes.axhspan(6, 8, alpha=0.15, color='#E63946', zorder=0)
        self.axes.axhspan(8, 10, alpha=0.15, color='#B71C1C', zorder=0)

    def _draw_legend(self):
        """绘制图例（位于标题右侧，与标题同一水平线）"""
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1B5E20', alpha=0.5, label='非常稳定 (0-2)'),
            Patch(facecolor='#4CAF50', alpha=0.5, label='稳定 (2-4)'),
            Patch(facecolor='#F4D03F', alpha=0.5, label='轻微波动 (4-6)'),
            Patch(facecolor='#E63946', alpha=0.5, label='不稳定 (6-8)'),
            Patch(facecolor='#B71C1C', alpha=0.5, label='情绪激烈 (8-10)')
        ]
        # 放置在图表顶部右侧，与标题同一水平线
        self.fig.legend(
            handles=legend_elements,
            loc='upper right',
            bbox_to_anchor=(0.98, 0.98),
            ncol=5,
            fontsize=7.5,
            frameon=True,
            framealpha=0.9,
            edgecolor='#000000',
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
                self.fig.set_facecolor('#F5F5DC')
                self.axes.set_facecolor('#F5F5DC')
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

            self.fig.set_facecolor('#F5F5DC')
            self.axes.set_facecolor('#F5F5DC')

            # 背景色带
            self._draw_background_zones()

            # 绘制趋势线
            self.axes.plot(x, y, color='#E63946', linewidth=2.5, marker='o',
                           markersize=6, markerfacecolor='#FFFFFF',
                           markeredgecolor='#E63946', markeredgewidth=2,
                           zorder=5)

            # 填充线下区域
            self.axes.fill_between(x, 0, y, alpha=0.08, color='#E63946', zorder=2)

            # 数据点标签（仅 <= 20 个时显示）
            if n <= 20:
                for xi, yi in zip(x, y):
                    self.axes.annotate(f'{yi:.1f}', (xi, yi), textcoords="offset points",
                                      xytext=(0, 10), ha='center', fontsize=8,
                                      fontweight='bold', color='#000000', zorder=6)

            # x 轴标签：使用日期时间
            self.axes.set_title("情绪稳定度变化趋势", fontsize=13, fontweight='bold', pad=12, loc='left')
            self.axes.set_ylabel("情绪稳定度 (0-10)", fontsize=10, fontweight='bold', labelpad=10)
            self.axes.set_ylim(-0.3, 10.3)
            self.axes.set_xlim(0.5, n + 0.5)
            self.axes.grid(True, alpha=0.3, linestyle='--', zorder=1)

            # 设置 x 轴刻度
            if n <= 15:
                # 少量数据：显示所有日期时间标签
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
                # 中等数据：每隔几个显示一个
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
                # 大量数据：只显示首尾和中间几个
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



class ToastNotification(QFrame):
    """
    苹果风格弹窗通知组件

    用于在界面右上角展示错误/警告信息，支持自动消失和手动关闭。
    """

    def __init__(self, parent, message, level="warning", duration=5000):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SubWindow)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedWidth(380)
        self._opacity = 1.0

        # 图标和颜色
        if level == "error":
            icon = "❌"
            bg_color = "#FDEDED"
            border_color = "#E63946"
            title_color = "#E63946"
            title_text = "错误"
        else:
            icon = "⚠️"
            bg_color = "#FFF8E1"
            border_color = "#F4D03F"
            title_color = "#B8860B"
            title_text = "警告"

        # 布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 12, 12)
        main_layout.setSpacing(10)

        # 图标
        icon_label = QLabel(icon)
        icon_label.setFont(QFont("Segoe UI Emoji", 16))
        icon_label.setFixedSize(32, 32)
        icon_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(icon_label)

        # 内容区
        content_layout = QVBoxLayout()
        content_layout.setSpacing(2)

        title_label = QLabel(title_text)
        title_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        title_label.setStyleSheet(f"color: {title_color}; background: transparent;")
        content_layout.addWidget(title_label)

        msg_label = QLabel(message)
        msg_label.setFont(QFont("Microsoft YaHei", 9))
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet("color: #4A4845; background: transparent;")
        msg_label.setMaximumWidth(280)
        content_layout.addWidget(msg_label)

        main_layout.addLayout(content_layout, 1)

        # 关闭按钮
        close_btn = QPushButton("×")
        close_btn.setFixedSize(24, 24)
        close_btn.setFont(QFont("Arial", 14))
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #000000;
                border-radius: 0px;
            }
            QPushButton:hover {
                background: rgba(0,0,0,0.1);
                color: #E63946;
            }
        """)
        close_btn.clicked.connect(self.fade_out)
        main_layout.addWidget(close_btn, 0, Qt.AlignTop)

        self.setStyleSheet(f"""
            ToastNotification {{
                background-color: {bg_color};
                border: 3px solid {border_color};
                border-radius: 0px;
            }}
        """)

        # 硬阴影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(0)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(3, 3)
        self.setGraphicsEffect(shadow)

        self.adjustSize()

        # 自动消失定时器
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.fade_out)
        self._timer.start(duration)

    def fade_out(self):
        """ 淡出动画后关闭 """
        self._timer.stop()
        # 用一个简单的定时器动画模拟淡出
        self._fade_timer = QTimer(self)
        self._fade_step = 0
        self._fade_timer.timeout.connect(self._do_fade)
        self._fade_timer.start(30)

    def _do_fade(self):
        self._fade_step += 1
        self._opacity = max(0, 1.0 - self._fade_step * 0.1)
        self.setWindowOpacity(self._opacity) if hasattr(self, 'setWindowOpacity') else None
        self.update()
        if self._fade_step >= 10:
            self._fade_timer.stop()
            self.close()
            self.deleteLater()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setOpacity(self._opacity)
        path = QPainterPath()
        path.addRect(0, 0, self.width(), self.height())
        painter.fillPath(path, QBrush(self.palette().window().color()))
        super().paintEvent(event)


class ToastManager:
    """
    Toast通知管理器

    管理多个Toast的显示位置，避免重叠。
    """
    def __init__(self, parent):
        self.parent = parent
        self.toasts = []

    def show_toast(self, message, level="warning", duration=5000):
        """ 显示一个Toast通知 """
        # 清理已关闭的toast
        self.toasts = [t for t in self.toasts if t.isVisible()]

        toast = ToastNotification(self.parent, message, level, duration)
        self.toasts.append(toast)

        # 计算位置（右上角堆叠）
        self._reposition()
        toast.show()

    def _reposition(self):
        """ 重新计算所有toast位置 """
        margin_top = 20
        margin_right = 20
        spacing = 10
        y_offset = margin_top

        parent_rect = self.parent.rect()
        for toast in self.toasts:
            if toast.isVisible():
                x = parent_rect.width() - toast.width() - margin_right
                toast.move(x, y_offset)
                y_offset += toast.height() + spacing


class ConstructivistBackground(QWidget):
    """构成主义几何背景层"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._cache = None

    def paintEvent(self, event):
        if self._cache and self._cache.size() == self.size():
            painter = QPainter(self)
            painter.drawPixmap(0, 0, self._cache)
            return

        from PyQt5.QtGui import QLinearGradient, QPixmap
        pixmap = QPixmap(self.size())
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.Antialiasing)

        # 渐变背景
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor("#F5F5DC"))
        gradient.setColorAt(1, QColor("#D3D3D3"))
        p.fillRect(self.rect(), gradient)


        # 角落几何色块装饰
        p.setPen(Qt.NoPen)
        # 左上红色三角
        p.setBrush(QColor("#E63946"))
        p.drawRect(0, 0, 35, 8)
        p.drawRect(0, 0, 8, 35)
        # 右下蓝色矩形
        p.setBrush(QColor("#1A237E"))
        p.drawRect(self.width()-45, self.height()-10, 45, 10)
        p.drawRect(self.width()-10, self.height()-45, 10, 45)
        # 右上黄色
        p.setBrush(QColor("#F4D03F"))
        p.drawRect(self.width()-30, 0, 30, 6)

        p.end()
        self._cache = pixmap
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._cache)

    def resizeEvent(self, event):
        self._cache = None
        super().resizeEvent(event)


class ScoreCard(QFrame):
    """
    分数卡片控件类

    自定义的分数展示卡片控件，用于在结果区域显示：
    - 情绪稳定度分数
    - 情绪状态等级
    - 主要情绪

    设计特点：
    - 构成主义风格：粗黑边框、直角、三角装饰
    - 大字号数字展示，醒目清晰
    - 支持自定义颜色和副标题
    - 支持重置为空状态

    参数：
        title (str): 卡片标题
        parent: 父窗口部件
        accent_color (str): 装饰色（红/黄/蓝）
    """
    def __init__(self, title, parent=None, accent_color="#E63946"):
        super().__init__(parent)
        self.setObjectName("scoreCard")
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._accent_color = accent_color

        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 22, 25, 22)
        layout.setSpacing(10)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("cardTitle")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
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

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        # 粗黑边框
        pen = QPen(QColor("#000000"))
        pen.setWidth(4)
        painter.setPen(pen)
        painter.drawRect(2, 2, self.width()-4, self.height()-4)
        # 左上角三角装饰
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(self._accent_color))
        from PyQt5.QtGui import QPolygon
        from PyQt5.QtCore import QPoint
        triangle = QPolygon([QPoint(0, 0), QPoint(24, 0), QPoint(0, 24)])
        painter.drawPolygon(triangle)
        painter.end()
        super().paintEvent(event)

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
        title_label.setStyleSheet("color: #000000; margin-bottom: 10px;")
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
                    border: 3px solid #000000;
                    border-radius: 0px;
                    padding: 8px 15px;
                    background-color: #F5F5DC;
                    color: #000000;
                    font-weight: bold;
                }
                QComboBox:hover {
                    border-color: #1A237E;
                }
                QComboBox:focus {
                    border-color: #1A237E;
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
                    background-color: #1A237E;
                    color: white;
                    border: 3px solid #000000;
                    border-radius: 0px;
                    padding: 12px 30px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #0D1652;
                }
                QPushButton:pressed {
                    background-color: #000051;
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
                    background-color: #1A237E;
                    color: white;
                    border: 3px solid #000000;
                    border-radius: 0px;
                    padding: 10px 30px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #0D1652;
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
        title_label.setStyleSheet("color: #000000; margin-bottom: 5px;")
        layout.addWidget(title_label)

        desc_label = QLabel("选择不同的模型会影响识别精度和运行速度，切换后需要重新加载模型。")
        desc_label.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        desc_label.setStyleSheet("color: #333333;")
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
            status_color = "#3D8C5C" if (is_current or downloaded) else "#B8860B"
            radio.setText(f"{display}  |  大小: {size}  |  {desc}")
            radio.setFont(QFont("Microsoft YaHei", 11))
            radio.setMinimumHeight(50)
            radio.setStyleSheet(f"""
                QRadioButton {{
                    padding: 12px 15px;
                    border: 3px solid {'#1A237E' if is_current else '#000000'};
                    border-radius: 0px;
                    background-color: {'#F5F5DC' if is_current else '#FFFFFF'};
                    font-weight: bold;
                }}
                QRadioButton:hover {{
                    border-color: #E63946;
                    background-color: #FFF8E1;
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
                background-color: #D3D3D3;
                color: #000000;
                border: 3px solid #000000;
                border-radius: 0px;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #BDBDBD;
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
                background-color: #1A237E;
                color: white;
                border: 3px solid #000000;
                border-radius: 0px;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0D1652;
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
        self.stats_label.setStyleSheet("padding: 12px; background-color: #F5F5DC; border-radius: 0px; border: 3px solid #000000; font-weight: bold;")
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

        self.dm_refresh_btn = QPushButton("🔄 刷新统计")
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
            self.dm_refresh_btn.setText("🔄 刷新中...")
            self.dm_refresh_btn.setEnabled(False)
            # 统计区域闪烁效果
            original_style = self.stats_label.styleSheet()
            self.stats_label.setStyleSheet("padding: 12px; background-color: #FFF8E1; border-radius: 0px; border: 3px solid #F4D03F; font-weight: bold;")
            from PyQt5.QtCore import QTimer
            def _restore():
                self.stats_label.setStyleSheet(original_style)
                self.refresh_stats()
                self.dm_refresh_btn.setText("🔄 刷新统计")
                self.dm_refresh_btn.setEnabled(True)
            QTimer.singleShot(300, _restore)
        except Exception as e:
            logger.error(f"刷新动画异常: {str(e)}")
            self.refresh_stats()
            self.dm_refresh_btn.setText("🔄 刷新统计")
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
                f"<a href='{path_to_url(app_dir)}' style='color: #1A237E;'>{app_dir}</a></p>"
                f"<p><b>数据目录（所有用户数据保存在此）：</b><br>"
                f"<a href='{path_to_url(data_dir)}' style='color: #1A237E;'>{data_dir}</a></p>"
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
            self.path_info.setText(path_html)
        except Exception as e:
            logger.error(f"刷新路径信息失败: {str(e)}")

    def refresh_stats(self):
        try:
            self.refresh_path_info()
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
            self._append_log_safe("✅ 情绪识别模型加载完成，准备就绪！")
            self.status_label.setText("● 准备就绪")
            self.status_label.setStyleSheet("color: #3D8C5C; background-color: #EDF7F0; padding: 8px 18px; border-radius: 2px; border: 3px solid #000000; font-weight: bold;")
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
            self.status_label.setStyleSheet("color: #E63946; background-color: #FDEDED; padding: 8px 18px; border-radius: 2px; border: 3px solid #000000; font-weight: bold;")
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

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")
        header_frame.setMinimumHeight(70)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(25, 12, 25, 12)

        title_label = QLabel("🎙️ 语音情绪识别系统")
        title_label.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        title_label.setObjectName("headerTitle")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        # 模型切换按钮
        self.model_switch_btn = QPushButton("🤖 切换模型")
        self.model_switch_btn.setFont(QFont("Microsoft YaHei", 10))
        self.model_switch_btn.setCursor(Qt.PointingHandCursor)
        self.model_switch_btn.setStyleSheet("""
            QPushButton {
                background-color: #F5F5DC;
                color: #1A237E;
                border: 3px solid #000000;
                border-radius: 2px;
                padding: 6px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1A237E;
                color: #FFFFFF;
                border-color: #000000;
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
        self.status_label.setStyleSheet("color: #B8860B; background-color: #FFF8E1; padding: 8px 18px; border-radius: 2px; border: 3px solid #000000; font-weight: bold;")

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
        result_layout.setSpacing(18)
        result_layout.setContentsMargins(25, 25, 25, 25)

        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(16)

        self.score_card = ScoreCard("情绪稳定度 (0-10)", accent_color="#E63946")
        self.level_card = ScoreCard("情绪状态", accent_color="#F4D03F")
        self.emotion_card = ScoreCard("主要情绪", accent_color="#1A237E")
        self.compound_card = ScoreCard("复合情绪", accent_color="#E63946")

        cards_layout.addWidget(self.score_card)
        cards_layout.addWidget(self.level_card)
        cards_layout.addWidget(self.emotion_card)
        cards_layout.addWidget(self.compound_card)

        result_layout.addLayout(cards_layout)

        # 分数说明
        stability_hint = QLabel("💡 情绪稳定度分数越低表示情绪越稳定，0分最稳定，10分波动最大")
        stability_hint.setFont(QFont("Microsoft YaHei", 10))
        stability_hint.setAlignment(Qt.AlignCenter)
        stability_hint.setStyleSheet("color: #000000; padding: 4px 0; font-weight: bold;")
        result_layout.addWidget(stability_hint)

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
        emotions = [("平静", "#8C9BA5"), ("开心", "#E5A83B"), ("惊讶", "#4DAFA0"),
                    ("悲伤", "#5B9BD5"), ("愤怒", "#D4554A"), ("恐惧", "#9B7DB8"),
                    ("厌恶", "#D4836B")]
        for emotion, color in emotions:
            bar_row = QHBoxLayout()
            bar_row.setSpacing(12)

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
            bar.setMinimumHeight(36)
            bar.setMaximumHeight(36)
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

        quick_guide = QGroupBox("📖 快速指南")
        quick_guide.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        quick_guide.setObjectName("guideGroup")
        guide_layout = QVBoxLayout(quick_guide)
        guide_layout.setContentsMargins(15, 25, 15, 15)
        guide_layout.setSpacing(10)

        # 使用流程
        steps_title = QLabel(" 使用流程")
        steps_title.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        steps_title.setStyleSheet("color: #E63946; padding: 2px 0; font-weight: bold;")
        guide_layout.addWidget(steps_title)

        steps_text = QLabel(
            "<div style='line-height: 2.0; font-size: 10.5pt; color: #000000;'>"
            "<div style='background: #F5F5DC; border-radius: 0px; padding: 8px 12px; margin: 4px 0; border-left: 4px solid #E63946;'>"
            "<b style='color: #E63946;'>1.</b> 等待模型加载完成（首次需下载模型）"
            "</div>"
            "<div style='background: #F5F5DC; border-radius: 0px; padding: 8px 12px; margin: 4px 0; border-left: 4px solid #F4D03F;'>"
            "<b style='color: #F4D03F;'>2.</b> 点击「开始录音」按钮"
            "</div>"
            "<div style='background: #F5F5DC; border-radius: 0px; padding: 8px 12px; margin: 4px 0; border-left: 4px solid #1A237E;'>"
            "<b style='color: #1A237E;'>3.</b> 自然地说出您的感受（建议 3-30 秒）"
            "</div>"
            "<div style='background: #F5F5DC; border-radius: 0px; padding: 8px 12px; margin: 4px 0; border-left: 4px solid #E63946;'>"
            "<b style='color: #E63946;'>4.</b> 点击「停止录音」按钮"
            "</div>"
            "<div style='background: #F5F5DC; border-radius: 0px; padding: 8px 12px; margin: 4px 0; border-left: 4px solid #F4D03F;'>"
            "<b style='color: #F4D03F;'>5.</b> 查看情绪分析结果（柱状图 + 情绪评分）"
            "</div>"
            "</div>"
        )
        steps_text.setWordWrap(True)
        steps_text.setTextFormat(Qt.RichText)
        guide_layout.addWidget(steps_text)

        # 功能说明
        features_title = QLabel("✨ 功能说明")
        features_title.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        features_title.setStyleSheet("color: #1A237E; padding: 6px 0 2px 0; font-weight: bold;")
        guide_layout.addWidget(features_title)

        features_text = QLabel(
            "<div style='line-height: 1.9; font-size: 10pt; color: #000000;'>"
            "• <b>多模型支持：</b>可切换 Large/Base/Seed 三种模型<br>"
            "• <b>7种情绪识别：</b>平静、开心、惊讶、悲伤、愤怒、恐惧、厌恶<br>"
            "• <b>复合情绪检测：</b>自动识别焦虑、挫败等复合情绪<br>"
            "• <b>情绪稳定度：</b>0-10分，分数越低越稳定<br>"
            "• <b>历史报告：</b>查看历史记录与情绪趋势图<br>"
            "• <b>数据管理：</b>清理录音、缓存等数据文件<br>"
            "• <b>便携模式：</b>所有数据保存在程序文件夹"
            "</div>"
        )
        features_text.setWordWrap(True)
        features_text.setTextFormat(Qt.RichText)
        guide_layout.addWidget(features_text)

        # 提示
        tip_label = QLabel(
            "<div style='background: #F5F5DC; border-radius: 0px; padding: 10px 12px; margin-top: 6px; font-size: 10pt; color: #000000; border: 2px solid #000000;'>"
            "<b>💡 小贴士：</b>在安静的环境下录音，效果会更好"
            "</div>"
        )
        tip_label.setWordWrap(True)
        tip_label.setTextFormat(Qt.RichText)
        guide_layout.addWidget(tip_label)

        right_layout.addWidget(quick_guide)

        content_splitter.addWidget(right_panel)

        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 1)
        content_splitter.setSizes([900, 300])

        layout.addWidget(content_splitter)
        self.tab_widget.addTab(realtime_widget, "🎙️ 实时检测")

    @exception_safe()
    def create_history_tab(self):
        history_widget = QWidget()
        layout = QVBoxLayout(history_widget)
        layout.setSpacing(12)

        history_btn_layout = QHBoxLayout()
        history_btn_layout.setSpacing(10)

        self.history_refresh_btn = QPushButton("🔄 刷新列表")
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
        self.history_list.setMinimumHeight(180)
        self.history_list.setMaximumHeight(280)
        self.history_list.setSpacing(2)
        self.history_list.setSelectionMode(QListWidget.NoSelection)
        list_layout.addWidget(self.history_list)
        splitter.addWidget(list_group)

        plot_group = QGroupBox("📈 情绪趋势图")
        plot_group.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        plot_group.setObjectName("plotGroup")
        plot_layout = QVBoxLayout(plot_group)
        plot_layout.setContentsMargins(15, 25, 15, 10)

        if MATPLOTLIB_AVAILABLE:
            self.canvas = MplCanvas(self, width=10, height=7, dpi=100)
            self.canvas.setMinimumHeight(350)
            plot_layout.addWidget(self.canvas)
        else:
            plot_placeholder = QLabel(
                "📊 绘图库未安装\n\n趋势图功能暂不可用"
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
        self.credit_label.setFont(QFont("Microsoft YaHei", 8, QFont.Bold))
        self.credit_label.setObjectName("creditLabel")
        self.credit_label.setWordWrap(True)
        bottom_container.addWidget(self.credit_label)

        parent_layout.addLayout(bottom_container)

    @exception_safe()
    def setup_styles(self):
        qss = """
        QMainWindow {
            background-color: #F5F5DC;
        }

        #headerFrame {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #F5F5DC, stop:1 #ECECEC);
            border: 3px solid #000000;
            border-radius: 0px;
        }

        #headerTitle {
            color: #000000;
            font-family: "Microsoft YaHei", "SimHei", "Arial Black";
            font-weight: bold;
        }

        #statusReady {
            color: #3D8C5C;
            background-color: #EDF7F0;
            padding: 8px 18px;
            border-radius: 2px;
            border: 3px solid #000000;
            font-weight: bold;
        }

        #statusLoading {
            color: #B8860B;
            background-color: #FDF6E3;
            padding: 8px 18px;
            border-radius: 2px;
            border: 3px solid #000000;
            font-weight: bold;
        }

        QMenuBar {
            background-color: #F5F5DC;
            border-bottom: 3px solid #000000;
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
            background-color: #1A237E;
            color: #FFFFFF;
        }

        QMenu {
            background-color: #F5F5DC;
            border: 3px solid #000000;
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
            background-color: #1A237E;
            color: #FFFFFF;
        }

        QTabWidget::pane {
            border: none;
            border-radius: 0px;
            background-color: transparent;
        }

        QTabBar::tab {
            background-color: #D3D3D3;
            padding: 13px 32px;
            margin-right: 4px;
            border-top-left-radius: 0px;
            border-top-right-radius: 0px;
            font-weight: bold;
            font-size: 13px;
            font-family: "Microsoft YaHei", "SimHei", "Arial Black";
            color: #000000;
            border: 3px solid #000000;
            border-bottom: none;
        }

        QTabBar::tab:selected {
            background-color: #F5F5DC;
            border-bottom: 3px solid #1A237E;
            color: #1A237E;
        }

        QTabBar::tab:hover:!selected {
            background-color: #F4D03F;
            color: #000000;
        }

        QGroupBox {
            border: 4px solid #000000;
            border-radius: 0px;
            margin-top: 12px;
            padding-top: 20px;
            background-color: rgba(245, 245, 220, 200);
            font-family: "Microsoft YaHei", "SimHei", "Arial Black";
            font-weight: bold;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 20px;
            padding: 0 12px;
            color: #000000;
            font-weight: bold;
        }

        #recordBtn {
            background-color: #C8C4BE;
            color: white;
            border: 3px solid #000000;
            border-radius: 2px;
            padding: 20px;
            font-family: "Microsoft YaHei", "SimHei", "Arial Black";
            font-weight: bold;
        }

        #recordBtn:enabled {
            background-color: #1A237E;
        }

        #recordBtn:enabled:hover {
            background-color: #0D1652;
        }

        #recordBtn:enabled:pressed {
            background-color: #000051;
        }

        #recordBtn[isRecording="true"] {
            background-color: #E63946;
        }

        #recordBtn[isRecording="true"]:hover {
            background-color: #B71C1C;
        }

        #durationLabel {
            background-color: #F5F5DC;
            border-radius: 0px;
            padding: 10px 20px;
            color: #000000;
            border: 3px solid #000000;
            font-weight: bold;
        }

        #recordProgress {
            border: 3px solid #000000;
            border-radius: 0px;
            background-color: #D3D3D3;
        }

        #recordProgress::chunk {
            background-color: #E63946;
            border-radius: 0px;
        }

        #scoreCard {
            background-color: rgba(245, 245, 220, 220);
            border-radius: 0px;
            border: 3px solid #000000;
        }

        #scoreCard:hover {
            border-color: #E63946;
        }

        #cardTitle {
            color: #000000;
            font-weight: bold;
            font-family: "Microsoft YaHei", "SimHei", "Arial Black";
        }

        #cardValue {
            color: #000000;
            font-family: "Microsoft YaHei", "SimHei", "Arial Black";
        }

        #cardSub {
            color: #333333;
            font-weight: bold;
        }

        #warningLabel {
            border-radius: 0px;
            padding: 12px 20px;
            border: 3px solid #000000;
        }

        QProgressBar {
            border: 3px solid #000000;
            border-radius: 0px;
            background-color: #D3D3D3;
            text-align: center;
            font-weight: bold;
            font-size: 10pt;
            font-family: "Microsoft YaHei", "SimHei", "Arial Black";
        }

        QProgressBar::chunk {
            border-radius: 0px;
            margin: 0px;
        }

        #probBar_平静::chunk { background-color: #8C9BA5; }
        #probBar_开心::chunk { background-color: #E5A83B; }
        #probBar_惊讶::chunk { background-color: #4DAFA0; }
        #probBar_悲伤::chunk { background-color: #5B9BD5; }
        #probBar_愤怒::chunk { background-color: #D4554A; }
        #probBar_恐惧::chunk { background-color: #9B7DB8; }
        #probBar_厌恶::chunk { background-color: #D4836B; }

        #historyList {
            border: 3px solid #000000;
            border-radius: 0px;
            background-color: #F5F5DC;
            padding: 8px;
            font-family: "Microsoft YaHei", "SimHei", "Arial Black";
            font-weight: bold;
        }

        #historyList::item {
            padding: 12px 15px;
            border-bottom: 2px solid #000000;
            border-radius: 0px;
            margin: 2px 0;
        }

        #historyList::item:selected {
            background-color: #1A237E;
            color: white;
        }

        #historyList::item:hover:!selected {
            background-color: #F4D03F;
            color: #000000;
        }

        #countLabel {
            color: #000000;
            padding: 5px 10px;
            font-weight: bold;
        }

        #dangerBtn {
            background-color: #E63946;
            color: white;
            border: 3px solid #000000;
            border-radius: 0px;
            padding: 8px 20px;
            font-weight: bold;
        }

        #dangerBtn:hover {
            background-color: #B71C1C;
        }

        QScrollArea {
            border: none;
            background-color: transparent;
        }

        QScrollBar:vertical {
            border: none;
            background: #D3D3D3;
            width: 10px;
            border-radius: 0px;
        }

        QScrollBar::handle:vertical {
            background: #000000;
            border-radius: 0px;
            min-height: 30px;
        }

        QScrollBar::handle:vertical:hover {
            background: #333333;
        }

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
        }

        #bottomBtnFrame {
            background-color: rgba(245, 245, 220, 220);
            border-radius: 0px;
            border: 3px solid #000000;
        }

        #dataBtn {
            background-color: #F4D03F;
            color: #000000;
            border: 3px solid #000000;
            border-radius: 0px;
            padding: 12px 28px;
            font-weight: bold;
            font-family: "Microsoft YaHei", "SimHei", "Arial Black";
        }

        #dataBtn:hover {
            background-color: #D4AC0D;
        }

        #creditLabel {
            color: #000000;
            padding: 5px;
            font-weight: bold;
        }

        QPushButton {
            outline: none;
            font-family: "Microsoft YaHei", "SimHei", "Arial Black";
            font-weight: bold;
        }

        QPushButton:disabled {
            background-color: #D3D3D3 !important;
            color: #666666 !important;
            cursor: not-allowed;
        }

        QSplitter::handle {
            background-color: #000000;
            width: 4px;
        }

        QSplitter::handle:hover {
            background-color: #E63946;
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
            self.status_label.setStyleSheet("color: #E63946; background-color: #FDEDED; padding: 8px 18px; border-radius: 2px; border: 3px solid #000000; font-weight: bold;")

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
            self.status_label.setStyleSheet("color: #B8860B; background-color: #FFF8E1; padding: 8px 18px; border-radius: 2px; border: 3px solid #000000; font-weight: bold;")
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
                self.status_label.setStyleSheet("color: #3D8C5C; background-color: #EDF7F0; padding: 8px 18px; border-radius: 2px; border: 3px solid #000000; font-weight: bold;")
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
        self.status_label.setStyleSheet("color: #E63946; background-color: #FDEDED; padding: 8px 18px; border-radius: 2px; border: 3px solid #000000; font-weight: bold;")
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
                        border: 4px solid {border_color};
                        border-radius: 0px;
                        margin-top: 12px;
                        padding-top: 20px;
                        background-color: {bg_color};
                    }}
                    QGroupBox::title {{
                        subcontrol-origin: margin;
                        left: 20px;
                        padding: 0 12px;
                        color: #000000;
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
                        self.warning_label.setText("⚠️ 情绪波动较大！建议立即进行深呼吸放松，必要时寻求亲友陪伴或专业帮助")
                        self.warning_label.setStyleSheet("""
                            background-color: #FDEDED;
                            color: #E63946;
                            border: 3px solid #E63946;
                            border-radius: 0px;
                            padding: 12px 20px;
                            font-weight: bold;
                        """)
                        self.status_label.setText("● 情绪不稳定")
                        self.status_label.setStyleSheet("color: #E63946; background-color: #FDEDED; padding: 8px 18px; border-radius: 2px; border: 3px solid #000000; font-weight: bold;")
                    else:
                        self.warning_label.setText("⚠️ 情绪存在一定波动，建议适当休息放松")
                        self.warning_label.setStyleSheet("""
                            background-color: #FFF8E1;
                            color: #B8860B;
                            border: 3px solid #F4D03F;
                            border-radius: 0px;
                            padding: 12px 20px;
                            font-weight: bold;
                        """)
                        self.status_label.setText("● 情绪波动")
                        self.status_label.setStyleSheet("color: #B8860B; background-color: #FFF8E1; padding: 8px 18px; border-radius: 2px; border: 3px solid #000000; font-weight: bold;")
                else:
                    self.warning_label.hide()
                    self.status_label.setText("● 情绪状态良好")
                    self.status_label.setStyleSheet("color: #3D8C5C; background-color: #EDF7F0; padding: 8px 18px; border-radius: 2px; border: 3px solid #000000; font-weight: bold;")

                self.append_log(f"分析完成 - 稳定度分数: {score:.1f}/10, 情绪状态: {level}, 主要情绪: {main_emotion}" + (f", 复合情绪: {compound_emotion}" if compound_emotion else ""))
                logger.info(f"分析完成 - 稳定度: {score:.1f}, 状态: {level}, 情绪: {main_emotion}" + (f", 复合情绪: {compound_emotion}" if compound_emotion else ""))
            else:
                error_msg = result.get('error', '未知错误')
                QMessageBox.warning(self, "分析错误", f"情绪分析失败:\n{error_msg}")
                self.append_log(f"分析错误: {error_msg}")
                logger.error(f"分析失败: {error_msg}")
                self.status_label.setText("● 分析失败")
                self.status_label.setStyleSheet("color: #E63946; background-color: #FDEDED; padding: 8px 18px; border-radius: 2px; border: 3px solid #000000; font-weight: bold;")
        except Exception as e:
            logger.error(f"处理分析结果异常: {str(e)}", exc_info=True)
            self.append_log(f"处理分析结果异常: {str(e)}")
            QMessageBox.critical(self, "错误", f"处理分析结果失败:\n{str(e)}")
            self.status_label.setText("● 处理错误")
            self.status_label.setStyleSheet("color: #E63946; background-color: #FDEDED; padding: 8px 18px; border-radius: 2px; border: 3px solid #000000; font-weight: bold;")

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
            self.history_refresh_btn.setText("🔄 刷新中...")
            self.history_refresh_btn.setEnabled(False)
            # 列表区域淡出效果
            self.history_list.setStyleSheet("opacity: 0.3; border: 3px solid #000000; border-radius: 0px; background-color: #D3D3D3; padding: 8px;")
            from PyQt5.QtCore import QTimer
            def _do_refresh():
                self.refresh_history()
                self.history_list.setStyleSheet("")
                self.history_refresh_btn.setText("🔄 刷新列表")
                self.history_refresh_btn.setEnabled(True)
            QTimer.singleShot(250, _do_refresh)
        except Exception as e:
            logger.error(f"刷新动画异常: {str(e)}")
            self.refresh_history()
            self.history_refresh_btn.setText("🔄 刷新列表")
            self.history_refresh_btn.setEnabled(True)

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
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_bg_widget'):
            self._bg_widget.setGeometry(self.centralWidget().rect())

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
