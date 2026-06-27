# -*- coding: utf-8 -*-
"""
录音流程测试

验证 AudioRecorder 基本功能和关键模块的导入。
需要麦克风硬件支持；若无可用音频设备，则自动跳过相关测试。
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _has_audio_device():
    """检查是否存在可用音频输入设备"""
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        n = pa.get_device_count()
        pa.terminate()
        return n > 0
    except Exception:
        return False


_HAS_AUDIO = _has_audio_device()


@unittest.skipUnless(_HAS_AUDIO, "未检测到音频输入设备，跳过录音流程测试")
class TestRecordingFlow(unittest.TestCase):
    """录音流程单元测试"""

    def test_recorder_init(self):
        from recorder import AudioRecorder
        recorder = AudioRecorder()
        self.assertIsNotNone(recorder)

    def test_start_and_stop_recording(self):
        from recorder import AudioRecorder
        recorder = AudioRecorder()
        temp_wav = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "portable_data", "temp", "test_recording.wav"
        )
        os.makedirs(os.path.dirname(temp_wav), exist_ok=True)

        success = recorder.start_recording(temp_wav)
        self.assertTrue(success, "录音启动失败")

        time.sleep(1.5)
        self.assertTrue(recorder.is_recording())

        output_path = recorder.stop_recording()
        self.assertTrue(output_path and os.path.exists(output_path),
                        "录音文件未生成")
        self.assertGreater(os.path.getsize(output_path), 1024,
                           "录音文件过小")

        # 清理
        try:
            os.remove(output_path)
        except Exception:
            pass


class TestModuleImports(unittest.TestCase):
    """关键模块导入测试"""

    def test_app_paths_import(self):
        from app_paths import get_app_dir, get_user_data_dir, get_recordings_dir
        self.assertIsInstance(get_app_dir(), str)

    def test_emotion_recognizer_import(self):
        from emotion_recognizer import EmotionRecognizer
        self.assertIsNotNone(EmotionRecognizer)

    def test_gui_import(self):
        try:
            from PyQt5.QtWidgets import QApplication
            from gui import MainWindow
            self.assertIsNotNone(QApplication)
            self.assertIsNotNone(MainWindow)
        except ImportError:
            self.skipTest("PyQt5 未安装，跳过 GUI 导入测试")


if __name__ == "__main__":
    unittest.main()
