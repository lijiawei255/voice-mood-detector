# -*- coding: utf-8 -*-
"""个人基线模块 — 单元测试"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pytest
import tempfile
from baseline import PersonalBaseline, get_baseline_path, MIN_BASELINE_SAMPLES


@pytest.fixture
def sample_result():
    return {
        "主要情绪": "平静", "置信度": 0.85, "情绪稳定度分数": 1.5,
        "valence_score": 0.3, "arousal_score": 0.15, "dominance_score": 0.75,
        "negative_load": 0.05, "emotional_uncertainty": 0.2,
        "完整概率_8类": {"愤怒": 0.02, "厌恶": 0.01, "恐惧": 0.02, "开心": 0.05,
                       "平静": 0.75, "其他": 0.05, "悲伤": 0.05, "惊讶": 0.05},
        "acoustic_features": {"f0_mean": 120, "f0_std": 15, "rms_mean": 0.08,
                            "rms_std": 0.02, "speech_rate": 3.5, "silence_ratio": 0.2,
                            "hnr_mean": 24, "jitter_local": 0.003, "shimmer_local": 0.01}
    }


class TestPersonalBaseline:
    def test_init_no_baseline(self):
        baseline = PersonalBaseline()
        assert not baseline.is_established()
        assert baseline.get_sample_count() == 0

    def test_add_samples_establishes_baseline(self, sample_result):
        baseline = PersonalBaseline()
        for i in range(MIN_BASELINE_SAMPLES):
            r = dict(sample_result)
            r["情绪稳定度分数"] = 1.5 + i * 0.1
            baseline.add_sample(r)
        assert baseline.is_established()
        bl = baseline.get_baseline()
        assert bl is not None
        assert "stability_mean" in bl
        assert bl["n_samples"] >= MIN_BASELINE_SAMPLES

    def test_compute_deviation(self, sample_result):
        baseline = PersonalBaseline()
        for i in range(MIN_BASELINE_SAMPLES):
            r = dict(sample_result)
            baseline.add_sample(r)

        # Test with a result significantly different from baseline
        test_result = dict(sample_result)
        test_result["情绪稳定度分数"] = 6.0
        test_result["valence_score"] = -0.5
        test_result["arousal_score"] = 0.8
        dev = baseline.compute_deviation(test_result)
        assert dev["available"]
        assert "deviations" in dev
        assert "personalized_stability_score" in dev
        assert dev["is_significant_deviation"]

    def test_reset(self, sample_result):
        baseline = PersonalBaseline()
        for i in range(MIN_BASELINE_SAMPLES):
            baseline.add_sample(sample_result)
        assert baseline.is_established()
        baseline.reset()
        assert not baseline.is_established()
        assert baseline.get_sample_count() == 0

    def test_deviation_without_baseline(self):
        baseline = PersonalBaseline()
        dev = baseline.compute_deviation({})
        assert not dev["available"]
