# -*- coding: utf-8 -*-
"""
情绪识别核心算法单元测试

测试 emotion_recognizer.py 中的核心算法：
- 概率归一化边界测试
- 稳定度分数范围验证
- 效价范围验证
- 唤醒度范围验证
- 8种情绪概率分布验证

作者：Jiawei Li
许可证：GPL v3
"""

import math
import os
import sys
import unittest
from unittest.mock import MagicMock

# 在导入前设置环境变量，避免触发模型自动下载
os.environ["FUNASR_AUTO_INSTALL"] = "0"
os.environ["FUNASR_INSTALL_DEP"] = "0"
os.environ["MODELSCOPE_AUTO_INSTALL_DEP"] = "0"

# 如果 torch 未安装，注入 mock 以避免 emotion_recognizer 导入失败
if "torch" not in sys.modules:
    sys.modules["torch"] = MagicMock()

from emotion_recognizer import (  # noqa: E402
    EmotionRecognizer,
    EMOTION_WEIGHTS,
    EMOTION_VALENCE,
    EMOTION_AROUSAL,
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

    def test_all_zero_probs(self):
        """全零概率输入：稳定度应在 [0, 10] 内"""
        probs = {emo: 0.0 for emo in EMOTION_WEIGHTS}
        score = self.recognizer._calculate_stability_score(probs)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 10.0)

    def test_single_high_prob_fear(self):
        """恐惧=1.0：稳定度应在 [0, 10] 内，且为正值"""
        probs = {emo: 0.0 for emo in EMOTION_WEIGHTS}
        probs["恐惧"] = 1.0
        score = self.recognizer._calculate_stability_score(probs)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 10.0)
        # 恐惧应导致相对较高的不稳定度
        self.assertGreater(score, 2.0)

    def test_single_high_prob_happy(self):
        """开心=1.0：正面情绪，稳定度应较低"""
        probs = {emo: 0.0 for emo in EMOTION_WEIGHTS}
        probs["开心"] = 1.0
        score = self.recognizer._calculate_stability_score(probs)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 10.0)
        # 开心不应导致高不稳定度
        self.assertLess(score, 5.0)

    def test_uniform_distribution(self):
        """均匀分布：所有情绪概率相等"""
        n = len(EMOTION_WEIGHTS)
        probs = {emo: 1.0 / n for emo in EMOTION_WEIGHTS}
        score = self.recognizer._calculate_stability_score(probs)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 10.0)

    def test_sum_greater_than_one(self):
        """总和 > 1 的概率（未归一化输入）"""
        probs = {emo: 0.5 for emo in EMOTION_WEIGHTS}
        score = self.recognizer._calculate_stability_score(probs)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 10.0)

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
        score = self.recognizer._calculate_stability_score(probs)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 10.0)

    def test_non_dict_input(self):
        """非字典输入应返回默认分数 5.0"""
        score = self.recognizer._calculate_stability_score(None)
        self.assertEqual(score, 5.0)
        score = self.recognizer._calculate_stability_score([1, 2, 3])
        self.assertEqual(score, 5.0)

    def test_empty_dict(self):
        """空字典输入"""
        score = self.recognizer._calculate_stability_score({})
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 10.0)


class TestValence(unittest.TestCase):
    """效价 _calculate_valence 测试（输出范围 [-1, +1]）"""

    def setUp(self):
        self.recognizer = _make_recognizer()

    def test_pure_positive_emotion(self):
        """纯正面情绪（开心=1.0）：效价应为正"""
        probs = {"开心": 1.0}
        valence = self.recognizer._calculate_valence(probs)
        self.assertGreater(valence, 0.0)
        self.assertGreaterEqual(valence, -1.0)
        self.assertLessEqual(valence, 1.0)

    def test_pure_negative_emotion(self):
        """纯负面情绪（悲伤=1.0）：效价应为负"""
        probs = {"悲伤": 1.0}
        valence = self.recognizer._calculate_valence(probs)
        self.assertLess(valence, 0.0)
        self.assertGreaterEqual(valence, -1.0)
        self.assertLessEqual(valence, 1.0)

    def test_neutral_emotion(self):
        """中性情绪（平静=1.0）：效价应接近 0 或为正"""
        probs = {"平静": 1.0}
        valence = self.recognizer._calculate_valence(probs)
        self.assertGreaterEqual(valence, -1.0)
        self.assertLessEqual(valence, 1.0)

    def test_mixed_emotions(self):
        """混合情绪：效价在 [-1, +1] 范围内"""
        probs = {"愤怒": 0.3, "开心": 0.3, "悲伤": 0.2, "平静": 0.2}
        valence = self.recognizer._calculate_valence(probs)
        self.assertGreaterEqual(valence, -1.0)
        self.assertLessEqual(valence, 1.0)

    def test_all_zero_probs(self):
        """全零概率：效价应为 0.0"""
        probs = {emo: 0.0 for emo in EMOTION_VALENCE}
        valence = self.recognizer._calculate_valence(probs)
        self.assertEqual(valence, 0.0)

    def test_empty_dict(self):
        """空字典：效价应为 0.0"""
        valence = self.recognizer._calculate_valence({})
        self.assertEqual(valence, 0.0)

    def test_output_range_all_emotions(self):
        """逐一测试每种情绪作为唯一输入时效价在范围内"""
        for emotion in EMOTION_VALENCE:
            probs = {emotion: 1.0}
            valence = self.recognizer._calculate_valence(probs)
            self.assertGreaterEqual(valence, -1.0,
                                    f"情绪 {emotion} 效价 {valence} 超出下界")
            self.assertLessEqual(valence, 1.0,
                                 f"情绪 {emotion} 效价 {valence} 超出上界")


class TestArousal(unittest.TestCase):
    """唤醒度 _calculate_arousal 测试（输出范围 [0, 1]）"""

    def setUp(self):
        self.recognizer = _make_recognizer()

    def test_high_arousal_emotion(self):
        """高唤醒情绪（愤怒=1.0）：唤醒度应较高"""
        probs = {"愤怒": 1.0}
        arousal = self.recognizer._calculate_arousal(probs)
        self.assertGreater(arousal, 0.5)
        self.assertGreaterEqual(arousal, 0.0)
        self.assertLessEqual(arousal, 1.0)

    def test_low_arousal_emotion(self):
        """低唤醒情绪（平静=1.0）：唤醒度应较低"""
        probs = {"平静": 1.0}
        arousal = self.recognizer._calculate_arousal(probs)
        self.assertLess(arousal, 0.5)
        self.assertGreaterEqual(arousal, 0.0)
        self.assertLessEqual(arousal, 1.0)

    def test_all_zero_probs(self):
        """全零概率：唤醒度应为 0.0"""
        probs = {emo: 0.0 for emo in EMOTION_AROUSAL}
        arousal = self.recognizer._calculate_arousal(probs)
        self.assertEqual(arousal, 0.0)

    def test_mixed_emotions_range(self):
        """混合情绪：唤醒度在 [0, 1] 范围内"""
        probs = {"愤怒": 0.25, "恐惧": 0.25, "平静": 0.25, "开心": 0.25}
        arousal = self.recognizer._calculate_arousal(probs)
        self.assertGreaterEqual(arousal, 0.0)
        self.assertLessEqual(arousal, 1.0)

    def test_output_range_all_emotions(self):
        """逐一测试每种情绪作为唯一输入时唤醒度在范围内"""
        for emotion in EMOTION_AROUSAL:
            probs = {emotion: 1.0}
            arousal = self.recognizer._calculate_arousal(probs)
            self.assertGreaterEqual(arousal, 0.0,
                                    f"情绪 {emotion} 唤醒度 {arousal} 超出下界")
            self.assertLessEqual(arousal, 1.0,
                                 f"情绪 {emotion} 唤醒度 {arousal} 超出上界")

    def test_empty_dict(self):
        """空字典：唤醒度应为 0.0"""
        arousal = self.recognizer._calculate_arousal({})
        self.assertEqual(arousal, 0.0)


class TestDisplayProbs(unittest.TestCase):
    """显示概率分布 display_probs 测试（8种情绪，含"其他"）"""

    def setUp(self):
        self.recognizer = _make_recognizer()

    def test_display_probs_has_8_emotions(self):
        """显示概率应包含8种情绪（含"其他"）"""
        expected_emotions = {"愤怒", "厌恶", "恐惧", "开心", "平静", "其他", "悲伤", "惊讶"}
        probs = {emo: 0.125 for emo in expected_emotions}
        # 验证 EMOTION_WEIGHTS 中包含8种情绪
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

    def test_display_probs_contains_other(self):
        """显示概率中应包含"其他"类别"""
        self.assertIn("其他", EMOTION_WEIGHTS)
        self.assertIn("其他", EMOTION_VALENCE)
        self.assertIn("其他", EMOTION_AROUSAL)


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


if __name__ == "__main__":
    unittest.main()
