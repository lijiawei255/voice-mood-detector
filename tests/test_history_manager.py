# -*- coding: utf-8 -*-
"""
历史记录管理器测试

覆盖 HistoryManager 中的：
- add_record
- 字段持久化
- get_statistics
- get_emotion_distribution
- get_stability_summary
- get_compound_emotion_stats
- 科研模式禁用自动清理

作者：Jiawei Li
许可证：GPL v3
"""

import os
import sys
import json
import tempfile
import shutil
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import history_manager
from history_manager import HistoryManager


class TestHistoryManager(unittest.TestCase):
    """HistoryManager 单元测试"""

    def setUp(self):
        """每个测试用例使用独立的临时数据目录"""
        self.temp_dir = tempfile.mkdtemp(prefix="voice_mood_test_")
        self.history_path = os.path.join(self.temp_dir, "history.json")

        # 通过 mock history_manager.get_history_path 让 HistoryManager 使用临时文件
        # 注意：history_manager.py 中使用的是 from app_paths import get_history_path，
        # 因此需要直接修改 history_manager 模块内的引用。
        self._orig_get_history_path = history_manager.get_history_path
        history_manager.get_history_path = lambda: self.history_path

        # 确保目录存在
        os.makedirs(self.temp_dir, exist_ok=True)
        self.manager = HistoryManager()

    def tearDown(self):
        """清理临时目录并恢复原始路径函数"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        history_manager.get_history_path = self._orig_get_history_path

    def _make_result(self, **overrides):
        """构造一条符合 add_record 输入的结果字典"""
        base = {
            "主要情绪": "平静",
            "置信度": 0.85,
            "情绪稳定度分数": 2.5,
            "情绪状态等级": "良好",
            "调节建议": "保持好心情",
            "情绪分析摘要": "测试摘要",
            "复合情绪": "",
            "复合情绪详情": None,
            "valence_score": 0.3,
            "arousal_score": 0.2,
            "dominance_score": 0.7,
            "negative_load": 0.1,
            "emotional_uncertainty": 0.2,
            "estimation_note": "测试说明",
            "完整概率_8类": {"愤怒": 0.05, "厌恶": 0.02, "恐惧": 0.03, "开心": 0.10,
                           "平静": 0.65, "其他": 0.05, "悲伤": 0.05, "惊讶": 0.05},
            "所有情绪概率": {"愤怒": 0.05, "厌恶": 0.02, "恐惧": 0.03, "开心": 0.10,
                           "平静": 0.65, "悲伤": 0.05, "惊讶": 0.05},
            "稳定度分项": {"negative_weight_score": 1.0, "entropy_score": 2.0,
                        "extremity_score": 0.5, "factor_weights_source": "测试"},
            "audio_quality": {"quality_score": 0.8, "quality_label": "良好"},
            "model_name": "emotion2vec_plus_large",
            "is_research_mode": False,
        }
        base.update(overrides)
        return base

    def test_add_record_success(self):
        """add_record 应成功添加记录并持久化"""
        result = self._make_result()
        self.assertTrue(self.manager.add_record(result))
        self.assertEqual(self.manager.get_record_count(), 1)

        # 验证文件已写入
        self.assertTrue(os.path.exists(self.history_path))
        with open(self.history_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)

    def test_record_fields_persisted(self):
        """关键字段应被正确保存和加载"""
        result = self._make_result(
            主要情绪="开心",
            情绪稳定度分数=1.5,
            复合情绪="焦虑",
        )
        self.manager.add_record(result)

        # 重新加载，验证字段
        new_manager = HistoryManager()
        records = new_manager.get_records()
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["main_emotion"], "开心")
        self.assertEqual(rec["anxiety_score"], 1.5)
        self.assertEqual(rec["emotion_level"], "良好")
        self.assertEqual(rec["compound_emotion"], "焦虑")
        self.assertIn("valence_score", rec)
        self.assertIn("arousal_score", rec)
        self.assertIn("dominance_score", rec)
        self.assertIn("negative_load", rec)
        self.assertIn("emotional_uncertainty", rec)
        self.assertIn("raw_model_output", rec)
        self.assertIn("probs_8", rec)
        self.assertIn("probs_7", rec)
        self.assertIn("stability_factors", rec)
        self.assertIn("audio_quality", rec)

    def test_get_statistics(self):
        """get_statistics 应返回正确的汇总统计"""
        self.manager.add_record(self._make_result(情绪稳定度分数=2.0, 置信度=0.8))
        self.manager.add_record(self._make_result(情绪稳定度分数=4.0, 置信度=0.6))
        self.manager.add_record(self._make_result(主要情绪="愤怒", 情绪稳定度分数=6.0, 置信度=0.7))

        stats = self.manager.get_statistics()
        self.assertEqual(stats["record_count"], 3)
        self.assertIn("stability", stats)
        self.assertIn("confidence", stats)
        self.assertIn("valence", stats)
        self.assertIn("arousal", stats)
        self.assertIn("dominance", stats)
        self.assertIn("emotion_distribution", stats)
        self.assertIn("compound_distribution", stats)

        # 验证分布
        dist = stats["emotion_distribution"]
        self.assertEqual(dist.get("平静"), 2)
        self.assertEqual(dist.get("愤怒"), 1)

    def test_get_emotion_distribution(self):
        """get_emotion_distribution 返回情绪频次"""
        self.manager.add_record(self._make_result(主要情绪="平静"))
        self.manager.add_record(self._make_result(主要情绪="开心"))
        self.manager.add_record(self._make_result(主要情绪="平静"))

        dist = self.manager.get_emotion_distribution()
        self.assertEqual(dist["平静"], 2)
        self.assertEqual(dist["开心"], 1)

    def test_get_stability_summary(self):
        """get_stability_summary 返回稳定度摘要"""
        self.manager.add_record(self._make_result(情绪稳定度分数=2.0))
        self.manager.add_record(self._make_result(情绪稳定度分数=3.0))
        self.manager.add_record(self._make_result(情绪稳定度分数=4.0))

        summary = self.manager.get_stability_summary()
        self.assertIn("mean", summary)
        self.assertIn("std", summary)
        self.assertIn("trend", summary)
        self.assertGreater(summary["mean"], 0)

    def test_get_compound_emotion_stats(self):
        """get_compound_emotion_stats 返回复合情绪频次"""
        self.manager.add_record(self._make_result(复合情绪="焦虑"))
        self.manager.add_record(self._make_result(复合情绪="焦虑"))
        self.manager.add_record(self._make_result(复合情绪="紧张"))

        stats = self.manager.get_compound_emotion_stats()
        self.assertEqual(stats["焦虑"], 2)
        self.assertEqual(stats["紧张"], 1)

    def test_research_mode_disables_auto_cleanup(self):
        """科研模式下不应触发自动清理"""
        # 降低阈值以便触发清理（通过直接修改类常量）
        original_threshold = HistoryManager.AUTO_CLEAN_THRESHOLD
        original_count = HistoryManager.AUTO_CLEAN_COUNT
        HistoryManager.AUTO_CLEAN_THRESHOLD = 5
        HistoryManager.AUTO_CLEAN_COUNT = 2

        cleanup_called = []

        def cleanup_callback(count):
            cleanup_called.append(count)

        try:
            # 非科研模式：添加超过阈值应触发清理
            # 使用独立临时文件，避免受 setUp 中 self.manager 的影响
            normal_path = os.path.join(self.temp_dir, "normal_history.json")
            history_manager.get_history_path = lambda: normal_path
            normal_manager = HistoryManager(cleanup_callback=cleanup_callback)
            for i in range(6):
                normal_manager.add_record(self._make_result(情绪稳定度分数=float(i)))
            self.assertGreater(len(cleanup_called), 0,
                               "非科研模式应触发自动清理回调")

            # 科研模式：不应触发清理
            research_path = os.path.join(self.temp_dir, "research_history.json")
            history_manager.get_history_path = lambda: research_path
            research_manager = HistoryManager()
            for i in range(6):
                research_manager.add_record(self._make_result(
                    情绪稳定度分数=float(i),
                    is_research_mode=True,
                ))
            # 科研模式下记录数应完整保留
            self.assertEqual(research_manager.get_record_count(), 6)
        finally:
            HistoryManager.AUTO_CLEAN_THRESHOLD = original_threshold
            HistoryManager.AUTO_CLEAN_COUNT = original_count
            # 恢复为默认的测试历史文件路径
            history_manager.get_history_path = lambda: self.history_path


if __name__ == "__main__":
    unittest.main()
