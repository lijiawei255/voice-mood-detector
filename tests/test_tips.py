# -*- coding: utf-8 -*-
"""
情绪调节建议系统测试

测试 relaxation_tips.py 中的 get_tips() 函数：
- 8 种基础情绪都能获取建议且 immediate 字段非空
- 高压力路径（anxiety_score >= 6.5）
- 低压力正面情绪返回正向反馈
- 兜底策略（未知情绪也能返回建议）
- 返回结构正确（immediate/long_term/compound_advice）
- 函数签名正确（无 arousal / sub_emotion 参数）

作者：Jiawei Li
许可证：GPL v3
"""

import inspect
import random
import unittest

from relaxation_tips import get_tips, TIPS, COMPOUND_TIPS

# 8 种基础情绪名称（与情绪识别器输出对齐）
BASE_EMOTIONS = ["愤怒", "厌恶", "恐惧", "悲伤", "开心", "平静", "惊讶", "其他"]


class TestEmotionTips(unittest.TestCase):
    """测试所有 8 种基础情绪都能获取建议"""

    def test_all_base_emotions_return_immediate_tips(self):
        """逐一测试每种基础情绪都能返回非空 immediate 建议"""
        random.seed(42)
        for emo in BASE_EMOTIONS:
            tips = get_tips(emo, anxiety_score=3.0)
            self.assertIsInstance(tips, dict,
                                  f"基础情绪 '{emo}' 返回类型不是 dict")
            self.assertIn("immediate", tips,
                          f"基础情绪 '{emo}' 返回缺少 immediate 字段")
            self.assertIn("long_term", tips,
                          f"基础情绪 '{emo}' 返回缺少 long_term 字段")
            self.assertIn("compound_advice", tips,
                          f"基础情绪 '{emo}' 返回缺少 compound_advice 字段")
            self.assertIsInstance(tips["immediate"], list,
                                  f"基础情绪 '{emo}' 的 immediate 不是 list")
            self.assertGreater(len(tips["immediate"]), 0,
                               f"基础情绪 '{emo}' 返回空 immediate 列表")

    def test_all_base_emotions_have_tip_entries(self):
        """验证负面情绪、高压力、平静、开心在 TIPS 字典中有对应条目"""
        # 注意："其他" 在 TIPS 中没有直接条目，会走兜底策略
        expected_in_tips = ["愤怒", "厌恶", "恐惧", "悲伤", "开心", "平静", "惊讶", "高压力"]
        for emo in expected_in_tips:
            self.assertIn(emo, TIPS,
                          f"情绪 '{emo}' 在 TIPS 字典中不存在")

    def test_no_sub_emotion_in_tips(self):
        """验证 TIPS 字典中不包含已移除的细分情绪"""
        removed = {
            "暴怒", "烦躁", "憎恶", "不满", "恐慌", "焦虑", "悲痛", "忧郁",
            "兴奋", "满足", "安详", "淡漠", "震惊", "好奇", "复杂", "平淡",
        }
        for key in removed:
            self.assertNotIn(key, TIPS,
                              f"已移除的细分情绪 '{key}' 仍存在于 TIPS 字典中")

    def test_get_tips_signature(self):
        """验证 get_tips() 函数签名与当前实现一致"""
        sig = inspect.signature(get_tips)
        param_names = list(sig.parameters.keys())
        self.assertNotIn("sub_emotion", param_names,
                         "get_tips() 仍包含 sub_emotion 参数，应该已移除")
        self.assertNotIn("arousal", param_names,
                         "get_tips() 仍包含 arousal 参数，应该已移除")
        self.assertEqual(param_names, ["emotion", "anxiety_score", "compound_emotion", "mixed_emotions"])


class TestHighStressPath(unittest.TestCase):
    """高压力路径（anxiety_score >= 6.5）返回包含'高压力'建议"""

    def test_high_stress_adds_stress_tips(self):
        """anxiety_score >= 6.5 时应包含 TIPS['高压力'] 中的建议"""
        random.seed(42)
        all_tips_seen = set()
        for _ in range(20):
            tips = get_tips("愤怒", anxiety_score=7.0)
            all_tips_seen.update(tips["immediate"])
        high_stress_tips = set(TIPS["高压力"])
        overlap = all_tips_seen & high_stress_tips
        self.assertGreater(len(overlap), 0,
                           "anxiety_score >= 6.5 时未返回任何'高压力'建议")

    def test_high_stress_boundary_exactly_6_5(self):
        """anxiety_score = 6.5 恰好触发高压力路径"""
        random.seed(42)
        all_tips_seen = set()
        for _ in range(20):
            tips = get_tips("恐惧", anxiety_score=6.5)
            all_tips_seen.update(tips["immediate"])
        high_stress_tips = set(TIPS["高压力"])
        overlap = all_tips_seen & high_stress_tips
        self.assertGreater(len(overlap), 0,
                           "anxiety_score = 6.5 时未触发高压力建议")

    def test_below_high_stress_no_stress_tips(self):
        """anxiety_score = 6.4 不应触发高压力建议"""
        random.seed(42)
        tips = get_tips("悲伤", anxiety_score=6.4)
        high_stress_tips = set(TIPS["高压力"])
        for tip in tips["immediate"]:
            self.assertNotIn(tip, high_stress_tips,
                             f"anxiety_score=6.4 时不应包含高压力建议，但找到了：{tip}")


class TestLowStressPositiveEmotion(unittest.TestCase):
    """低压力正面情绪返回正向反馈"""

    def test_positive_emotion_happy(self):
        """开心 + 低压力：返回正向反馈"""
        random.seed(42)
        tips = get_tips("开心", anxiety_score=2.0)
        self.assertIsInstance(tips, dict)
        self.assertGreater(len(tips["immediate"]), 0)
        # 不应包含高压力建议
        high_stress_tips = set(TIPS["高压力"])
        for tip in tips["immediate"]:
            self.assertNotIn(tip, high_stress_tips)

    def test_positive_emotion_calm(self):
        """平静 + 低压力：返回正向反馈"""
        random.seed(42)
        tips = get_tips("平静", anxiety_score=1.0)
        self.assertIsInstance(tips, dict)
        self.assertGreater(len(tips["immediate"]), 0)

    def test_positive_emotion_neutral_and_surprised(self):
        """中性/负面情绪低压力时也能返回建议"""
        random.seed(42)
        for emo in ["平静", "惊讶", "其他", "厌恶"]:
            tips = get_tips(emo, anxiety_score=2.0)
            self.assertGreater(len(tips["immediate"]), 0,
                               f"情绪 '{emo}' + 低压力时未返回建议")


class TestCompoundEmotionPath(unittest.TestCase):
    """复合情绪建议路径测试"""

    def test_compound_emotion_priority(self):
        """检测到复合情绪时，结果中应包含复合情绪专属建议"""
        random.seed(42)
        tips = get_tips("恐惧", anxiety_score=5.0, compound_emotion="焦虑")
        self.assertIsInstance(tips, dict)
        self.assertGreater(len(tips["immediate"]), 0)
        self.assertIsInstance(tips["compound_advice"], str)
        self.assertGreater(len(tips["compound_advice"]), 0,
                           "复合情绪 '焦虑' 应返回非空 compound_advice")

    def test_invalid_compound_emotion_ignored(self):
        """无效复合情绪名称应被忽略，不影响基础建议返回"""
        random.seed(42)
        tips = get_tips("愤怒", anxiety_score=5.0, compound_emotion="不存在")
        self.assertIsInstance(tips, dict)
        self.assertGreater(len(tips["immediate"]), 0)
        self.assertEqual(tips["compound_advice"], "")


class TestFallbackStrategy(unittest.TestCase):
    """兜底策略（未知情绪也能返回建议）"""

    def test_unknown_emotion_low_stress(self):
        """未知情绪 + 低压力：应返回默认正向反馈"""
        random.seed(42)
        tips = get_tips("未知情绪", anxiety_score=2.0)
        self.assertIsInstance(tips, dict)
        self.assertGreater(len(tips["immediate"]), 0,
                           "未知情绪 + 低压力时兜底策略未返回建议")

    def test_unknown_emotion_high_stress(self):
        """未知情绪 + 高压力：应返回高压力建议"""
        random.seed(42)
        tips = get_tips("未知情绪", anxiety_score=8.0)
        self.assertIsInstance(tips, dict)
        self.assertGreater(len(tips["immediate"]), 0,
                           "未知情绪 + 高压力时兜底策略未返回建议")
        high_stress_tips = set(TIPS["高压力"])
        self.assertTrue(
            any(tip in high_stress_tips for tip in tips["immediate"]),
            "未知情绪 + 高压力时应至少包含一条高压力建议"
        )

    def test_empty_string_emotion(self):
        """空字符串情绪：仍应返回建议"""
        random.seed(42)
        tips = get_tips("", anxiety_score=3.0)
        self.assertIsInstance(tips, dict)
        self.assertGreater(len(tips["immediate"]), 0,
                           "空字符串情绪时兜底策略未返回建议")


class TestReturnStructure(unittest.TestCase):
    """返回结构正确性测试"""

    def test_length_between_1_and_3(self):
        """大量调用验证 immediate 长度始终在 [1, 3] 范围内"""
        random.seed(42)
        test_cases = [
            ("愤怒", 7.0),
            ("开心", 2.0),
            ("平静", 1.0),
            ("恐惧", 6.5),
            ("悲伤", 4.0),
            ("未知情绪", 5.0),
            ("其他", 3.0),
            ("惊讶", 2.0),
            ("厌恶", 5.5),
        ]
        for emotion, score in test_cases:
            for _ in range(10):
                tips = get_tips(emotion, score)
                immediate = tips["immediate"]
                self.assertGreaterEqual(len(immediate), 1,
                                        f"immediate 为空: emotion={emotion}, score={score}")
                self.assertLessEqual(len(immediate), 3,
                                     f"immediate 超过3条: emotion={emotion}, score={score}")

    def test_long_term_provided_above_threshold(self):
        """anxiety_score >= 3.5 时应可能返回长期建议"""
        random.seed(42)
        tips = get_tips("愤怒", anxiety_score=4.0)
        self.assertIsInstance(tips["long_term"], list)

    def test_compound_advice_type(self):
        """compound_advice 字段类型为 str"""
        tips = get_tips("开心", anxiety_score=2.0)
        self.assertIsInstance(tips["compound_advice"], str)


if __name__ == "__main__":
    unittest.main()
