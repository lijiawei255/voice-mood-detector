# -*- coding: utf-8 -*-
"""
数据导出模块测试

覆盖 export_manager.py 中的：
- export_records_csv
- export_research_dataset
- export_session_report

作者：Jiawei Li
许可证：GPL v3
"""

import os
import sys
import json
import csv
import tempfile
import shutil
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from export_manager import (
    export_records_csv,
    export_research_dataset,
    export_session_report,
)


class TestExportManager(unittest.TestCase):
    """数据导出模块单元测试"""

    def setUp(self):
        """每个测试用例使用独立的临时目录"""
        self.temp_dir = tempfile.mkdtemp(prefix="voice_mood_export_test_")

    def tearDown(self):
        """清理临时目录"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_record(self, **overrides):
        """构造一条历史记录"""
        base = {
            "id": "20240101_120000_abcdef",
            "timestamp": "2024-01-01 12:00:00",
            "main_emotion": "平静",
            "confidence": 0.85,
            "anxiety_score": 2.5,
            "emotion_level": "良好",
            "suggestion": "保持好心情",
            "compound_emotion": "",
            "compound_emotion_detail": {},
            "emotion_summary": "测试摘要",
            "valence_score": 0.3,
            "arousal_score": 0.2,
            "dominance_score": 0.7,
            "negative_load": 0.1,
            "emotional_uncertainty": 0.2,
            "estimation_note": "测试说明",
            "stability_factors": {
                "negative_weight_score": 1.0,
                "entropy_score": 2.0,
                "extremity_score": 0.5,
            },
            "audio_quality": {
                "quality_score": 0.8,
                "quality_label": "良好",
                "speech_ratio": 0.7,
                "rms_mean": 0.1,
                "noise_level": 0.01,
                "clipping_ratio": 0.0,
                "issues": [],
            },
            "model_name": "emotion2vec_plus_large",
            "algorithm_version": "2.0.0-p0",
            "app_version": "2.0.0",
            "is_research_mode": False,
            "assessment_reliability": "高",
        }
        base.update(overrides)
        return base

    def test_export_records_csv_success(self):
        """CSV 导出应成功并生成正确表头"""
        records = [
            self._make_record(main_emotion="平静", anxiety_score=2.5),
            self._make_record(main_emotion="开心", anxiety_score=1.5),
        ]
        output_path = os.path.join(self.temp_dir, "history.csv")
        self.assertTrue(export_records_csv(records, output_path))
        self.assertTrue(os.path.exists(output_path))

        with open(output_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # 表头 + 2 行数据
        self.assertEqual(len(rows), 3)
        header = rows[0]
        self.assertIn("id", header)
        self.assertIn("main_emotion", header)
        self.assertIn("stability_score", header)
        self.assertIn("valence_score", header)
        self.assertIn("arousal_score", header)
        self.assertIn("dominance_score", header)
        self.assertIn("negative_weight_score", header)
        self.assertIn("entropy_score", header)
        self.assertIn("extremity_score", header)

    def test_export_records_csv_compound_description(self):
        """CSV compound_emotion_description 应读取 detail['desc'] 而非 'description'"""
        records = [
            self._make_record(
                compound_emotion="焦虑",
                compound_emotion_detail={"name": "焦虑", "desc": "恐惧与悲伤的复合情绪", "confidence": 0.6},
            ),
        ]
        output_path = os.path.join(self.temp_dir, "compound.csv")
        self.assertTrue(export_records_csv(records, output_path))

        with open(output_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 1)
        # 关键修复点：desc 键必须被正确导出，不能为空
        self.assertEqual(rows[0]["compound_emotion"], "焦虑")
        self.assertEqual(rows[0]["compound_emotion_description"], "恐惧与悲伤的复合情绪")

    def test_export_records_csv_empty(self):
        """空记录列表应导出失败"""
        output_path = os.path.join(self.temp_dir, "empty.csv")
        self.assertFalse(export_records_csv([], output_path))
        self.assertFalse(os.path.exists(output_path))

    def test_export_research_dataset_success(self):
        """研究数据集 JSON 导出应成功并包含完整记录"""
        records = [
            self._make_record(main_emotion="恐惧", anxiety_score=7.0),
            self._make_record(main_emotion="愤怒", anxiety_score=6.5),
        ]
        output_path = os.path.join(self.temp_dir, "dataset.json")
        self.assertTrue(export_research_dataset(records, output_path))
        self.assertTrue(os.path.exists(output_path))

        with open(output_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        self.assertIn("export_timestamp", dataset)
        self.assertIn("record_count", dataset)
        self.assertIn("schema_version", dataset)
        self.assertIn("records", dataset)
        self.assertEqual(dataset["record_count"], 2)
        self.assertEqual(len(dataset["records"]), 2)

    def test_export_research_dataset_empty(self):
        """空记录列表应导出失败"""
        output_path = os.path.join(self.temp_dir, "empty.json")
        self.assertFalse(export_research_dataset([], output_path))
        self.assertFalse(os.path.exists(output_path))

    def test_export_session_report_success(self):
        """会话报告导出应成功"""
        result = {
            "主要情绪": "平静",
            "置信度": 0.85,
            "情绪稳定度分数": 2.5,
            "情绪状态等级": "良好",
            "稳定度分项": {
                "negative_weight_score": 1.0,
                "entropy_score": 2.0,
                "extremity_score": 0.5,
            },
            "valence_score": 0.3,
            "arousal_score": 0.2,
            "dominance_score": 0.7,
            "negative_load": 0.1,
            "emotional_uncertainty": 0.2,
            "estimation_note": "测试说明",
            "复合情绪详情": None,
            "完整概率_8类": {"愤怒": 0.05, "厌恶": 0.02, "恐惧": 0.03, "开心": 0.10,
                           "平静": 0.65, "其他": 0.05, "悲伤": 0.05, "惊讶": 0.05},
            "所有情绪概率": {"愤怒": 0.05, "厌恶": 0.02, "恐惧": 0.03, "开心": 0.10,
                           "平静": 0.65, "悲伤": 0.05, "惊讶": 0.05},
            "audio_quality": {"quality_score": 0.8},
            "调节建议": "保持好心情",
            "情绪分析摘要": "测试摘要",
        }
        output_path = os.path.join(self.temp_dir, "report.json")
        self.assertTrue(export_session_report(result, output_path))
        self.assertTrue(os.path.exists(output_path))

        with open(output_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertIn("report_timestamp", report)
        self.assertIn("assessment", report)
        self.assertIn("vad_dimensions", report)
        self.assertIn("compound_emotion", report)
        self.assertIn("probabilities_8class", report)
        self.assertIn("metadata", report)
        self.assertIn("suggestion", report)
        self.assertIn("summary", report)


if __name__ == "__main__":
    unittest.main()
