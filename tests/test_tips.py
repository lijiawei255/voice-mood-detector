# -*- coding: utf-8 -*-
"""
情绪调节建议系统测试

测试 relaxation_tips.py 中的 get_tips() 函数：
- 8 种基础情绪都能获取建议且每种至少 3 条
- 高压力路径（anxiety_score >= 6）
- 低压力正面情绪返回正向反馈
- 高唤醒条件触发
- 低唤醒条件触发
- 兜底策略（未知情绪也能返回建议）
- 返回列表长度在 1-3 之间
- 函数签名正确（无 sub_emotion 参数）

作者：Jiawei Li
许可证：GPL v3
"""

import inspect
import random
import unittest

from relaxation_tips import get_tips, TIPS

# 8 种基础情绪名称
BASE_EMOTIONS = ["愤怒", "厌恶", "恐惧", "悲伤", "开心", "平静", "惊讶", "其他"]


class TestEmotionTips(unittest.TestCase):
    """测试所有 8 种基础情绪都能获取建议"""

    def test_all_base_emotions_return_tips(self):
        """逐一测试每种基础情绪都能返回非空建议列表"""
        random.seed(42)
        for emo in BASE_EMOTIONS:
            tips = get_tips(emo, anxiety_score=3.0)
            self.assertIsInstance(tips, list,
                                  f"基础情绪 '{emo}' 返回类型不是 list")
            self.assertGreater(len(tips), 0,
                               f"基础情绪 '{emo}' 返回空列表")

    def test_all_base_emotions_have_tip_entries(self):
        """验证所有 8 种基础情绪都在 TIPS 字典中有对应条目"""
        for emo in BASE_EMOTIONS:
            self.assertIn(emo, TIPS,
                          f"基础情绪 '{emo}' 在 TIPS 字典中不存在")
            self.assertGreaterEqual(
                len(TIPS[emo]), 3,
                f"基础情绪 '{emo}' 的建议只有 {len(TIPS[emo])} 条，需要至少 3 条",
            )

    def test_no_sub_emotion_in_tips(self):
        """验证 TIPS 字典中不包含已移除的 16 种细分情绪"""
        removed = {
            "暴怒", "烦躁", "憎恶", "不满", "恐慌", "焦虑", "悲痛", "忧郁",
            "兴奋", "满足", "安详", "淡漠", "震惊", "好奇", "复杂", "平淡",
        }
        for key in removed:
            self.assertNotIn(key, TIPS,
                              f"已移除的细分情绪 '{key}' 仍存在于 TIPS 字典中")

    def test_get_tips_no_sub_emotion_param(self):
        """验证 get_tips() 函数签名中不再有 sub_emotion 参数"""
        sig = inspect.signature(get_tips)
        param_names = list(sig.parameters.keys())
        self.assertNotIn("sub_emotion", param_names,
                         "get_tips() 仍包含 sub_emotion 参数，应该已移除")
        # 验证期望的参数顺序
        self.assertEqual(param_names, ["emotion", "anxiety_score", "arousal"])


class TestHighStressPath(unittest.TestCase):
    """高压力路径（anxiety_score >= 6）返回包含'高压力'建议"""

    def test_high_stress_adds_stress_tips(self):
        """anxiety_score >= 6 时应包含 TIPS['高压力'] 中的建议"""
        random.seed(42)
        all_tips_seen = set()
        for _ in range(20):
            tips = get_tips("愤怒", anxiety_score=7.0)
            all_tips_seen.update(tips)
        high_stress_tips = set(TIPS["高压力"])
        overlap = all_tips_seen & high_stress_tips
        self.assertGreater(len(overlap), 0,
                           "anxiety_score >= 6 时未返回任何'高压力'建议")

    def test_high_stress_boundary_exactly_6(self):
        """anxiety_score = 6.0 恰好触发高压力路径"""
        random.seed(42)
        all_tips_seen = set()
        for _ in range(20):
            tips = get_tips("恐惧", anxiety_score=6.0)
            all_tips_seen.update(tips)
        high_stress_tips = set(TIPS["高压力"])
        overlap = all_tips_seen & high_stress_tips
        self.assertGreater(len(overlap), 0,
                           "anxiety_score = 6.0 时未触发高压力建议")

    def test_below_high_stress_no_stress_tips(self):
        """anxiety_score = 5.9 不应触发高压力建议"""
        random.seed(42)
        tips = get_tips("悲伤", anxiety_score=5.9)
        high_stress_tips = set(TIPS["高压力"])
        for tip in tips:
            self.assertNotIn(tip, high_stress_tips,
                             f"anxiety_score=5.9 时不应包含高压力建议，但找到了：{tip}")


class TestLowStressPositiveEmotion(unittest.TestCase):
    """低压力正面情绪返回正向反馈"""

    def test_positive_emotion_happy(self):
        """开心 + 低压力：返回正向反馈"""
        random.seed(42)
        tips = get_tips("开心", anxiety_score=2.0)
        self.assertIsInstance(tips, list)
        self.assertGreater(len(tips), 0)
        # 不应包含高压力建议
        high_stress_tips = set(TIPS["高压力"])
        for tip in tips:
            self.assertNotIn(tip, high_stress_tips)

    def test_positive_emotion_calm(self):
        """平静 + 低压力：返回正向反馈"""
        random.seed(42)
        tips = get_tips("平静", anxiety_score=1.0)
        self.assertIsInstance(tips, list)
        self.assertGreater(len(tips), 0)

    def test_positive_emotion_neutral_and_surprised(self):
        """中性情绪（平静/惊讶/其他）低压力时也能返回建议"""
        random.seed(42)
        for emo in ["平静", "惊讶", "其他"]:
            tips = get_tips(emo, anxiety_score=2.0)
            self.assertGreater(len(tips), 0,
                               f"情绪 '{emo}' + 低压力时未返回建议")


class TestArousalConditions(unittest.TestCase):
    """高唤醒和低唤醒条件触发测试"""

    def test_high_arousal_triggers(self):
        """arousal > 0.7 且 anxiety_score >= 4 时应包含高唤醒建议"""
        random.seed(42)
        all_tips_seen = set()
        for _ in range(20):
            tips = get_tips("愤怒", anxiety_score=5.0, arousal=0.8)
            all_tips_seen.update(tips)
        high_arousal_tips = set(TIPS["高唤醒"])
        overlap = all_tips_seen & high_arousal_tips
        self.assertGreater(len(overlap), 0,
                           "arousal > 0.7 且 anxiety_score >= 4 时未触发高唤醒建议")

    def test_high_arousal_not_triggered_low_anxiety(self):
        """arousal > 0.7 但 anxiety_score < 4 不应触发高唤醒建议"""
        random.seed(42)
        tips = get_tips("开心", anxiety_score=2.0, arousal=0.8)
        high_arousal_tips = set(TIPS["高唤醒"])
        for tip in tips:
            self.assertNotIn(tip, high_arousal_tips,
                             f"anxiety_score < 4 时不应包含高唤醒建议，但找到了：{tip}")

    def test_low_arousal_triggers(self):
        """arousal < 0.3 且 anxiety_score < 4 时应包含低唤醒建议"""
        random.seed(42)
        all_tips_seen = set()
        for _ in range(20):
            tips = get_tips("平静", anxiety_score=1.0, arousal=0.2)
            all_tips_seen.update(tips)
        low_arousal_tips = set(TIPS["低唤醒"])
        overlap = all_tips_seen & low_arousal_tips
        self.assertGreater(len(overlap), 0,
                           "arousal < 0.3 且 anxiety_score < 4 时未触发低唤醒建议")

    def test_low_arousal_not_triggered_high_anxiety(self):
        """arousal < 0.3 但 anxiety_score >= 4 不应触发低唤醒建议"""
        random.seed(42)
        tips = get_tips("悲伤", anxiety_score=5.0, arousal=0.2)
        low_arousal_tips = set(TIPS["低唤醒"])
        for tip in tips:
            self.assertNotIn(tip, low_arousal_tips,
                             f"anxiety_score >= 4 时不应包含低唤醒建议，但找到了：{tip}")


class TestFallbackStrategy(unittest.TestCase):
    """兜底策略（未知情绪也能返回建议）"""

    def test_unknown_emotion_low_stress(self):
        """未知情绪 + 低压力：应返回默认正向反馈"""
        random.seed(42)
        tips = get_tips("未知情绪", anxiety_score=2.0)
        self.assertIsInstance(tips, list)
        self.assertGreater(len(tips), 0,
                           "未知情绪 + 低压力时兜底策略未返回建议")

    def test_unknown_emotion_high_stress(self):
        """未知情绪 + 高压力：应返回高压力建议"""
        random.seed(42)
        tips = get_tips("未知情绪", anxiety_score=8.0)
        self.assertIsInstance(tips, list)
        self.assertGreater(len(tips), 0,
                           "未知情绪 + 高压力时兜底策略未返回建议")

    def test_empty_string_emotion(self):
        """空字符串情绪：仍应返回建议"""
        random.seed(42)
        tips = get_tips("", anxiety_score=3.0)
        self.assertIsInstance(tips, list)
        self.assertGreater(len(tips), 0,
                           "空字符串情绪时兜底策略未返回建议")


class TestReturnLength(unittest.TestCase):
    """返回列表长度在 1-3 之间"""

    def test_length_between_1_and_3(self):
        """大量调用验证返回长度始终在 [1, 3] 范围内"""
        random.seed(42)
        test_cases = [
            ("愤怒", 7.0, 0.8),
            ("开心", 2.0, 0.5),
            ("平静", 1.0, 0.1),
            ("恐惧", 6.0, 0.9),
            ("悲伤", 4.0, 0.3),
            ("未知情绪", 5.0, None),
            ("其他", 3.0, 0.4),
            ("惊讶", 2.0, 0.5),
            ("厌恶", 5.5, 0.6),
        ]
        for emotion, score, arousal in test_cases:
            for _ in range(10):
                tips = get_tips(emotion, score, arousal=arousal)
                self.assertGreaterEqual(len(tips), 1,
                                        f"建议列表为空: emotion={emotion}, score={score}")
                self.assertLessEqual(len(tips), 3,
                                     f"建议列表超过3条: emotion={emotion}, score={score}")


if __name__ == "__main__":
    unittest.main()
