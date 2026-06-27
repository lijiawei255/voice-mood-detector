# -*- coding: utf-8 -*-
"""
后台工作线程模块

提供独立的 QThread 子类，用于在后台执行耗时操作，避免阻塞 GUI：
- RecordingThread: 音频录制线程
- AnalysisThread: AI 模型推理线程

（模型加载由 EmotionRecognizer.load_model 内部异步完成，
 通过 model_loaded_signal 信号回主线程，故无需独立的 ModelLoadThread。）

作者：Jiawei Li
许可证：GPL v3
"""

import os
import logging

from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


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
