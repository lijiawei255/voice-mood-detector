# -*- coding: utf-8 -*-
"""
端到端识别测试

使用已下载的模型对 tests/portable_data/recordings/ 或 tests/ 目录下的 WAV 文件
进行实际推理测试。若模型未下载或无测试音频，则自动跳过。

注意：本测试文件改用 pytest 风格，避免在导入阶段阻塞模型加载。
"""

import os
import sys
import unittest

# 在导入前设置环境变量
os.environ['FUNASR_AUTO_INSTALL'] = '0'
os.environ['FUNASR_INSTALL_DEP'] = '0'
os.environ['MODELSCOPE_AUTO_INSTALL_DEP'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app_paths import setup_modelscope_cache, is_model_downloaded, load_model_config

setup_modelscope_cache()


def _get_test_wav_files():
    """获取测试目录中的 WAV 文件"""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(test_dir, "portable_data", "recordings"),
        os.path.join(os.path.dirname(test_dir), "portable_data", "recordings"),
        test_dir,
    ]
    files = []
    for directory in candidates:
        if os.path.isdir(directory):
            for f in os.listdir(directory):
                if f.lower().endswith('.wav'):
                    files.append(os.path.join(directory, f))
    return list(set(files))


_MODEL_NAME = load_model_config()
_MODEL_DOWNLOADED = is_model_downloaded(_MODEL_NAME)
_WAV_FILES = _get_test_wav_files()


@unittest.skipUnless(_MODEL_DOWNLOADED, "未检测到已下载模型，跳过端到端识别测试")
class TestEndToEndRecognition(unittest.TestCase):
    """端到端推理测试"""

    @classmethod
    def setUpClass(cls):
        from emotion_recognizer import EmotionRecognizer
        cls.recognizer = EmotionRecognizer(model_name=_MODEL_NAME)
        import threading
        loaded_event = threading.Event()
        cls._load_error = None

        def on_loaded(success, error):
            if not success:
                cls._load_error = error
            loaded_event.set()

        cls.recognizer.load_model(callback=on_loaded)
        loaded_event.wait(timeout=120)
        if not cls.recognizer.is_ready():
            raise unittest.SkipTest(f"模型加载失败或超时: {cls._load_error or '未知错误'}")

    @unittest.skipUnless(len(_WAV_FILES) > 0, "未找到 WAV 测试文件")
    def test_predict_returns_valid_result(self):
        """对找到的测试音频运行推理并验证返回结构"""
        for audio_path in _WAV_FILES[:3]:
            with self.subTest(file=os.path.basename(audio_path)):
                result = self.recognizer.predict(audio_path)
                self.assertIsInstance(result, dict)
                self.assertIn("success", result)
                if result.get("success"):
                    self.assertIn("主要情绪", result)
                    self.assertIn("置信度", result)
                    self.assertIn("情绪稳定度分数", result)
                    self.assertGreaterEqual(result["置信度"], 0.0)
                    self.assertLessEqual(result["置信度"], 1.0)


if __name__ == "__main__":
    unittest.main()
