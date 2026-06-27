# -*- coding: utf-8 -*-
"""
可靠性评估模块测试

覆盖 reliability.py 中的：
- compute_icc
- evaluate_model_agreement
- evaluate_multi_sample_reliability
- get_reliability_level

作者：Jiawei Li
许可证：GPL v3
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from reliability import (
    compute_icc,
    evaluate_model_agreement,
    evaluate_multi_sample_reliability,
    get_reliability_level,
    MODEL_AGREEMENT_NOTE,
)


class TestComputeICC(unittest.TestCase):
    """测试-重测信度 ICC 计算"""

    def test_high_reliability(self):
        """组间差异大、组内高度一致的数据应返回较高 ICC"""
        values = [
            [2.0, 2.05, 1.98],
            [8.0, 8.02, 8.05],
            [5.0, 5.03, 4.97],
        ]
        icc = compute_icc(values)
        self.assertGreater(icc, 0.8)
        self.assertLessEqual(icc, 1.0)

    def test_low_reliability(self):
        """组间差异小、组内差异大时应返回较低 ICC"""
        values = [
            [1.0, 9.0, 2.0],
            [2.0, 8.0, 3.0],
            [1.5, 9.5, 2.5],
        ]
        icc = compute_icc(values)
        self.assertLess(icc, 0.3)
        self.assertGreaterEqual(icc, -1.0)

    def test_insufficient_sessions(self):
        """session 不足时应返回 0.0"""
        self.assertEqual(compute_icc([]), 0.0)
        self.assertEqual(compute_icc([[1.0]]), 0.0)

    def test_icc_range(self):
        """ICC 值应在 [-1, 1] 范围内"""
        values = [
            [3.0, 4.0, 3.5],
            [3.2, 4.1, 3.6],
            [2.9, 3.8, 3.4],
        ]
        icc = compute_icc(values)
        self.assertGreaterEqual(icc, -1.0)
        self.assertLessEqual(icc, 1.0)


class TestEvaluateModelAgreement(unittest.TestCase):
    """双模型一致性（模型规模敏感性）测试"""

    def test_empty_results(self):
        """空输入应返回低一致性"""
        result = evaluate_model_agreement(None, {})
        self.assertEqual(result["agreement"], 0.0)
        self.assertEqual(result["level"], "低")
        self.assertIn("note", result)

    def test_perfect_agreement(self):
        """完全相同的结果应返回高一致性"""
        r = {
            "主要情绪": "开心",
            "情绪稳定度分数": 2.5,
            "valence_score": 0.5,
            "arousal_score": 0.4,
            "dominance_score": 0.7,
            "model_name": "base",
        }
        result = evaluate_model_agreement(r, r)
        self.assertGreaterEqual(result["agreement"], 0.8)
        self.assertEqual(result["level"], "高")
        self.assertTrue(result["emotion_match"])

    def test_emotion_mismatch(self):
        """主要情绪不一致时应降低一致性"""
        r1 = {
            "主要情绪": "开心",
            "情绪稳定度分数": 2.5,
            "valence_score": 0.5,
            "arousal_score": 0.4,
            "dominance_score": 0.7,
        }
        r2 = dict(r1)
        r2["主要情绪"] = "愤怒"
        r2["valence_score"] = -0.5
        result = evaluate_model_agreement(r1, r2)
        self.assertFalse(result["emotion_match"])
        self.assertLess(result["agreement"], 1.0)

    def test_note_present(self):
        """返回结果应包含模型一致性说明"""
        result = evaluate_model_agreement({}, {})
        self.assertIn(MODEL_AGREEMENT_NOTE, result["note"])


class TestEvaluateMultiSampleReliability(unittest.TestCase):
    """多次采样综合可靠性评估测试"""

    def test_empty_list(self):
        """空列表应返回低可靠性"""
        result = evaluate_multi_sample_reliability([])
        self.assertEqual(result["consistency"], 0.0)
        self.assertEqual(result["reliability"], "低")

    def test_single_sample(self):
        """单条样本应返回基本统计"""
        result = evaluate_multi_sample_reliability([
            {"主要情绪": "平静", "情绪稳定度分数": 2.0,
             "valence_score": 0.3, "arousal_score": 0.2}
        ])
        self.assertEqual(result["n_samples"], 1)
        self.assertEqual(result["session_emotion"], "平静")
        self.assertEqual(result["emotion_consistency"], 1.0)

    def test_high_consistency(self):
        """稳定一致的多条样本应返回高可靠性"""
        samples = [
            {"主要情绪": "平静", "情绪稳定度分数": 2.1,
             "valence_score": 0.30, "arousal_score": 0.20},
            {"主要情绪": "平静", "情绪稳定度分数": 2.2,
             "valence_score": 0.31, "arousal_score": 0.21},
            {"主要情绪": "平静", "情绪稳定度分数": 2.0,
             "valence_score": 0.29, "arousal_score": 0.19},
        ]
        result = evaluate_multi_sample_reliability(samples)
        self.assertEqual(result["session_emotion"], "平静")
        self.assertGreaterEqual(result["consistency"], 0.7)
        self.assertEqual(result["reliability"], "高")

    def test_low_consistency(self):
        """波动大的多条样本应返回低可靠性"""
        samples = [
            {"主要情绪": "开心", "情绪稳定度分数": 1.0,
             "valence_score": 0.8, "arousal_score": 0.1},
            {"主要情绪": "愤怒", "情绪稳定度分数": 8.0,
             "valence_score": -0.8, "arousal_score": 0.9},
        ]
        result = evaluate_multi_sample_reliability(samples)
        self.assertLess(result["consistency"], 0.7)

    def test_consistency_range(self):
        """consistency 应在 [0, 1] 范围内"""
        samples = [
            {"主要情绪": "平静", "情绪稳定度分数": 3.0,
             "valence_score": 0.0, "arousal_score": 0.5},
        ]
        result = evaluate_multi_sample_reliability(samples)
        self.assertGreaterEqual(result["consistency"], 0.0)
        self.assertLessEqual(result["consistency"], 1.0)


class TestGetReliabilityLevel(unittest.TestCase):
    """综合可靠性等级评定测试"""

    def test_high_reliability(self):
        """高质量 + 高置信度 + 高一致性 = 高"""
        level = get_reliability_level({
            "audio_quality": {"quality_score": 0.9},
            "confidence": 0.9,
            "consistency": 0.9,
        })
        self.assertEqual(level, "高")

    def test_medium_reliability(self):
        """中等分数 = 中"""
        level = get_reliability_level({
            "audio_quality": {"quality_score": 0.5},
            "confidence": 0.5,
        })
        self.assertEqual(level, "中")

    def test_low_reliability(self):
        """低质量 + 低置信度 = 低"""
        level = get_reliability_level({
            "audio_quality": {"quality_score": 0.1},
            "confidence": 0.1,
        })
        self.assertEqual(level, "低")

    def test_all_zero_scores_returns_low(self):
        """所有有效分数都很低时应返回低"""
        self.assertEqual(get_reliability_level({
            "audio_quality": {"quality_score": 0.0},
            "confidence": 0.0,
        }), "低")


if __name__ == "__main__":
    unittest.main()
