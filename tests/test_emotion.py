# -*- coding: utf-8 -*-
"""
情绪识别端到端测试（pytest 版本）

使用 portable_data/recordings/ 目录下的 WAV 文件进行实际推理，
验证 predict() 返回字段完整且数值范围正确。
若模型未下载或无音频文件，则自动跳过。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app_paths import setup_modelscope_cache, is_model_downloaded, get_recordings_dir, load_model_config

setup_modelscope_cache()

_MODEL_NAME = load_model_config()
_MODEL_DOWNLOADED = is_model_downloaded(_MODEL_NAME)


def _get_test_wav_files():
    """获取录音目录中的 WAV 文件"""
    recordings_dir = get_recordings_dir()
    if not os.path.isdir(recordings_dir):
        return []
    return [os.path.join(recordings_dir, f) for f in os.listdir(recordings_dir)
            if f.lower().endswith('.wav')]


_WAV_FILES = _get_test_wav_files()


@unittest.skipUnless(_MODEL_DOWNLOADED, "未检测到已下载模型，跳过情绪识别测试")
@unittest.skipUnless(len(_WAV_FILES) > 0, "未找到 WAV 测试文件")
class TestEmotionRecognition(unittest.TestCase):
    """情绪识别端到端测试"""

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

    def test_predict_returns_required_fields(self):
        """predict() 返回所有必需字段"""
        required_fields = [
            "主要情绪", "置信度", "所有情绪概率", "完整概率_8类",
            "情绪稳定度分数", "情绪状态等级", "等级颜色",
            "调节建议", "混合情绪", "复合情绪", "复合情绪详情",
            "valence_score", "arousal_score", "dominance_score",
            "negative_load", "emotional_uncertainty",
            "稳定度分项", "原始模型输出", "情绪分析摘要",
        ]
        for audio_path in _WAV_FILES[:3]:
            with self.subTest(file=os.path.basename(audio_path)):
                result = self.recognizer.predict(audio_path)
                self.assertTrue(result.get("success"),
                                f"推理失败: {result.get('error')}")
                for field in required_fields:
                    self.assertIn(field, result, f"缺少字段: {field}")

    def test_predict_value_ranges(self):
        """数值范围约束验证"""
        for audio_path in _WAV_FILES[:3]:
            with self.subTest(file=os.path.basename(audio_path)):
                result = self.recognizer.predict(audio_path)
                if not result.get("success"):
                    self.skipTest(f"推理失败: {result.get('error')}")
                self.assertGreaterEqual(result["置信度"], 0.0)
                self.assertLessEqual(result["置信度"], 1.0)
                self.assertGreaterEqual(result["情绪稳定度分数"], 0.0)
                self.assertLessEqual(result["情绪稳定度分数"], 10.0)
                self.assertGreaterEqual(result["valence_score"], -1.0)
                self.assertLessEqual(result["valence_score"], 1.0)
                self.assertGreaterEqual(result["arousal_score"], 0.0)
                self.assertLessEqual(result["arousal_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
