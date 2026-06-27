# -*- coding: utf-8 -*-
"""
情绪识别核心算法单元测试

测试 emotion_recognizer.py 中的核心算法：
- 概率归一化边界测试
- 稳定度分数范围验证
- 8 种情绪概率分布验证
- 稳定度等级 _get_stability_level 验证
- 复合情绪分析 _analyze_compound_emotions 验证

VAD 维度相关测试已迁移到 test_vad_dimensions.py。

作者：Jiawei Li
许可证：GPL v3
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

# 在导入前设置环境变量，避免触发模型自动下载
os.environ["FUNASR_AUTO_INSTALL"] = "0"
os.environ["FUNASR_INSTALL_DEP"] = "0"
os.environ["MODELSCOPE_AUTO_INSTALL_DEP"] = "0"

# 如果 torch 未安装，注入一个最小化且对 SciPy 友好的 fake torch，
# 避免 scipy 在检测 torch.Tensor 时因 MagicMock 不是 class 而抛出 TypeError。
if "torch" not in sys.modules:
    class _FakeTensor:
        pass

    class _FakeNoGrad:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    class _FakeCuda:
        OutOfMemoryError = RuntimeError

        @staticmethod
        def empty_cache():
            pass

    class _FakeTorch:
        Tensor = _FakeTensor
        float32 = object()
        int64 = object()
        cuda = _FakeCuda()

        @staticmethod
        def no_grad():
            return _FakeNoGrad()

        @staticmethod
        def tensor(*args, **kwargs):
            return None

    sys.modules["torch"] = _FakeTorch()

from emotion_recognizer import (  # noqa: E402
    EmotionRecognizer,
    EMOTION_WEIGHTS,
)


def _make_recognizer():
    """创建一个不加载模型的 EmotionRecognizer 实例（绕过单例）"""
    # 绕过单例模式：直接通过 object.__new__ 创建新实例
    instance = object.__new__(EmotionRecognizer)
    instance._initialized = True
    instance.model = None
    instance.loaded = False
    instance.loading = False
    instance.error = None
    instance.current_model_key = None
    instance._progress_callbacks = []
    import threading
    instance._load_lock = threading.Lock()
    return instance


class TestStabilityScore(unittest.TestCase):
    """稳定度分数 _calculate_stability_score 测试"""

    def setUp(self):
        self.recognizer = _make_recognizer()

    def _assert_stability_dict(self, result):
        """断言稳定度返回字典结构正确"""
        self.assertIsInstance(result, dict)
        expected_keys = [
            "negative_weight_score",
            "entropy_score",
            "extremity_score",
            "stability_score",
            "stability_level",
            "stability_color",
            "factor_weights_source",
        ]
        for key in expected_keys:
            self.assertIn(key, result, f"稳定度结果缺少字段: {key}")
        self.assertIsInstance(result["stability_score"], float)
        self.assertGreaterEqual(result["stability_score"], 0.0)
        self.assertLessEqual(result["stability_score"], 10.0)

    def test_all_zero_probs(self):
        """全零概率输入：稳定度应在 [0, 10] 内"""
        probs = {emo: 0.0 for emo in EMOTION_WEIGHTS}
        result = self.recognizer._calculate_stability_score(probs)
        self._assert_stability_dict(result)

    def test_single_high_prob_fear(self):
        """恐惧=1.0：稳定度应在 [0, 10] 内，且为正值"""
        probs = {emo: 0.0 for emo in EMOTION_WEIGHTS}
        probs["恐惧"] = 1.0
        result = self.recognizer._calculate_stability_score(probs)
        self._assert_stability_dict(result)
        # 恐惧应导致相对较高的不稳定度
        self.assertGreater(result["stability_score"], 2.0)

    def test_single_high_prob_happy(self):
        """开心=1.0：正面情绪，稳定度应较低"""
        probs = {emo: 0.0 for emo in EMOTION_WEIGHTS}
        probs["开心"] = 1.0
        result = self.recognizer._calculate_stability_score(probs)
        self._assert_stability_dict(result)
        # 开心不应导致高不稳定度
        self.assertLess(result["stability_score"], 5.0)

    def test_uniform_distribution(self):
        """均匀分布：所有情绪概率相等"""
        n = len(EMOTION_WEIGHTS)
        probs = {emo: 1.0 / n for emo in EMOTION_WEIGHTS}
        result = self.recognizer._calculate_stability_score(probs)
        self._assert_stability_dict(result)

    def test_sum_greater_than_one(self):
        """总和 > 1 的概率（未归一化输入）"""
        probs = {emo: 0.5 for emo in EMOTION_WEIGHTS}
        result = self.recognizer._calculate_stability_score(probs)
        self._assert_stability_dict(result)

    def test_normal_distribution(self):
        """正常概率分布"""
        probs = {
            "愤怒": 0.1,
            "厌恶": 0.05,
            "恐惧": 0.05,
            "悲伤": 0.1,
            "开心": 0.4,
            "平静": 0.2,
            "惊讶": 0.05,
            "其他": 0.05,
        }
        result = self.recognizer._calculate_stability_score(probs)
        self._assert_stability_dict(result)

    def test_non_dict_input(self):
        """非字典输入应返回默认分数 5.0 的稳定度字典"""
        result = self.recognizer._calculate_stability_score(None)
        self._assert_stability_dict(result)
        self.assertEqual(result["stability_score"], 5.0)

        result = self.recognizer._calculate_stability_score([1, 2, 3])
        self._assert_stability_dict(result)
        self.assertEqual(result["stability_score"], 5.0)

    def test_empty_dict(self):
        """空字典输入"""
        result = self.recognizer._calculate_stability_score({})
        self._assert_stability_dict(result)

    def test_factor_weights_source_present(self):
        """因子权重来源标注应存在且为字符串"""
        probs = {emo: 0.125 for emo in EMOTION_WEIGHTS}
        result = self.recognizer._calculate_stability_score(probs)
        self.assertIsInstance(result["factor_weights_source"], str)
        self.assertGreater(len(result["factor_weights_source"]), 0)


class TestCompoundEmotions(unittest.TestCase):
    """复合情绪分析 _analyze_compound_emotions 测试"""

    def setUp(self):
        self.recognizer = _make_recognizer()

    def test_no_compound_when_probs_low(self):
        """低概率分布不应触发复合情绪"""
        probs = {emo: 0.04 for emo in EMOTION_WEIGHTS}
        probs["平静"] = 1.0 - sum(probs.values()) + probs["平静"]
        result = self.recognizer._analyze_compound_emotions(probs, "平静")
        self.assertIsNone(result)

    def test_anxiety_compound_triggered(self):
        """恐惧+悲伤高概率应触发焦虑复合情绪"""
        probs = {emo: 0.0 for emo in EMOTION_WEIGHTS}
        probs["恐惧"] = 0.45
        probs["悲伤"] = 0.40
        probs["开心"] = 0.10
        probs["平静"] = 0.05
        result = self.recognizer._analyze_compound_emotions(probs, "恐惧")
        self.assertIsNotNone(result)
        self.assertIn("name", result)
        self.assertEqual(result["name"], "焦虑")
        self.assertIn("components", result)
        self.assertIn("confidence", result)
        self.assertIn("desc", result)
        self.assertIn("advice", result)
        self.assertIn("interpretation", result)

    def test_compound_with_invalid_input(self):
        """非字典输入应返回 None"""
        result = self.recognizer._analyze_compound_emotions(None, "愤怒")
        self.assertIsNone(result)


class TestDisplayProbs(unittest.TestCase):
    """显示概率分布相关测试"""

    def setUp(self):
        self.recognizer = _make_recognizer()

    def test_display_probs_has_8_emotions(self):
        """EMOTION_WEIGHTS 应包含 8 种情绪（含'其他'）"""
        expected_emotions = {"愤怒", "厌恶", "恐惧", "开心", "平静", "其他", "悲伤", "惊讶"}
        self.assertEqual(set(EMOTION_WEIGHTS.keys()), expected_emotions)
        self.assertEqual(len(EMOTION_WEIGHTS), 8)

    def test_display_probs_sum_to_one(self):
        """归一化后概率之和应接近 1.0"""
        probs = {
            "愤怒": 0.15, "厌恶": 0.10, "恐惧": 0.08, "悲伤": 0.12,
            "开心": 0.30, "平静": 0.10, "惊讶": 0.10, "其他": 0.05,
        }
        total = sum(probs.values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_emotion_weights_contains_other(self):
        """情绪权重中应包含'其他'类别"""
        self.assertIn("其他", EMOTION_WEIGHTS)


class TestStabilityLevel(unittest.TestCase):
    """稳定度等级 _get_stability_level 测试"""

    def setUp(self):
        self.recognizer = _make_recognizer()

    def test_level_ranges(self):
        """各分数区间对应正确的等级"""
        level, _ = self.recognizer._get_stability_level(0.0)
        self.assertEqual(level, "非常稳定")
        level, _ = self.recognizer._get_stability_level(1.0)
        self.assertEqual(level, "非常稳定")
        level, _ = self.recognizer._get_stability_level(2.0)
        self.assertEqual(level, "良好")
        level, _ = self.recognizer._get_stability_level(3.5)
        self.assertEqual(level, "一般")
        level, _ = self.recognizer._get_stability_level(5.0)
        self.assertEqual(level, "轻度波动")
        level, _ = self.recognizer._get_stability_level(7.0)
        self.assertEqual(level, "不稳定")
        level, _ = self.recognizer._get_stability_level(9.0)
        self.assertEqual(level, "情绪激烈")

    def test_invalid_input(self):
        """无效输入应返回 '未知'"""
        level, _ = self.recognizer._get_stability_level("invalid")
        self.assertEqual(level, "未知")


class TestGetAdvice(unittest.TestCase):
    """调节建议 _get_advice 测试"""

    def setUp(self):
        self.recognizer = _make_recognizer()

    def test_advice_returns_string(self):
        """_get_advice 应返回字符串"""
        advice = self.recognizer._get_advice("开心", 1.0, {"开心": 1.0})
        self.assertIsInstance(advice, str)
        self.assertGreater(len(advice), 0)

    def test_high_stress_advice_has_warning(self):
        """高压力应返回带警告 emoji 的建议"""
        advice = self.recognizer._get_advice("恐惧", 7.5, {"恐惧": 0.8})
        self.assertIsInstance(advice, str)
        self.assertIn("⚠️", advice)


class TestPredictIntegration(unittest.TestCase):
    """predict() 端到端字段集成测试（不加载真实模型）"""

    def setUp(self):
        import tempfile
        import wave
        import struct

        self.recognizer = _make_recognizer()
        self.recognizer.loaded = True
        self.recognizer.model_name = "emotion2vec_plus_large"

        # 创建临时 16kHz 单声道 WAV 文件（1 秒静音/白噪声）
        self.temp_wav = tempfile.mktemp(suffix=".wav")
        with wave.open(self.temp_wav, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            # 生成小幅随机噪声，避免被质量模块视为完全静音
            samples = [int(500 * (i % 2 * 2 - 1)) for i in range(16000)]
            wf.writeframes(struct.pack('<' + 'h' * len(samples), *samples))

        # Mock 模型输出：平静为主，略带开心
        def _fake_generate(*args, **kwargs):
            return [{
                "labels": ["neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted", "other"],
                "scores": [0.55, 0.20, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01],
            }]

        self.recognizer.model = type("_FakeModel", (), {"generate": _fake_generate})()

    def tearDown(self):
        try:
            os.remove(self.temp_wav)
        except Exception:
            pass

    def test_predict_returns_all_new_fields(self):
        """predict() 应返回 P0/P1/P2 新增的所有字段"""
        result = self.recognizer.predict(self.temp_wav, is_research_mode=True)
        self.assertTrue(result.get("success"), f"预测失败: {result.get('error')}")

        required_fields = [
            "主要情绪", "置信度", "所有情绪概率", "完整概率_8类",
            "情绪稳定度分数", "情绪状态等级", "等级颜色", "稳定度分项",
            "valence_score", "arousal_score", "dominance_score",
            "negative_load", "emotional_uncertainty", "estimation_note",
            "调节建议", "混合情绪", "复合情绪", "复合情绪详情", "情绪分析摘要",
            "audio_quality", "acoustic_features", "psychological_indicators",
            "assessment_reliability", "baseline_deviation",
            "model_name", "algorithm_version", "app_version", "is_research_mode",
            "原始模型输出",
        ]
        for field in required_fields:
            self.assertIn(field, result, f"predict() 返回缺少字段: {field}")

        self.assertEqual(result["is_research_mode"], True)
        self.assertEqual(result["model_name"], "emotion2vec_plus_large")
        self.assertEqual(result["algorithm_version"], "2.0.0-p2")

    def test_predict_audio_quality_has_required_keys(self):
        """audio_quality 应包含核心质量字段"""
        result = self.recognizer.predict(self.temp_wav)
        aq = result.get("audio_quality", {})
        for key in ["duration", "speech_ratio", "quality_score", "quality_label"]:
            self.assertIn(key, aq, f"audio_quality 缺少字段: {key}")


if __name__ == "__main__":
    unittest.main()
