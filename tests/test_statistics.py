# -*- coding: utf-8 -*-
"""统计分析模块 — 单元测试"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pytest
from statistics import (
    cohens_d, hedges_g, descriptive_stats, detect_trend,
    detect_anomalies, detect_high_volatility_events,
    compare_emotion_distributions, estimate_mcid,
)


class TestCohensD:
    def test_large_effect(self):
        g1 = [1, 2, 3, 2, 1]
        g2 = [8, 9, 10, 9, 8]
        result = cohens_d(g1, g2)
        assert result["d"] > 1.5
        assert result["magnitude"] == "large"

    def test_small_effect(self):
        g1 = [5, 6, 5, 6, 5]
        g2 = [5, 7, 5, 7, 6]
        result = cohens_d(g1, g2)
        assert abs(result["d"]) < 1.0

    def test_paired(self):
        g1 = [2.0, 3.5, 4.0, 3.0, 2.5]
        g2 = [3.0, 4.5, 5.5, 4.0, 3.0]
        result = cohens_d(g1, g2, paired=True)
        assert result["d"] > 0

    def test_small_sample(self):
        result = cohens_d([1], [2])
        assert result["interpretation"] == "样本量不足"


class TestHedgesG:
    def test_g_approx_d_large_n(self):
        g1 = list(range(30))
        g2 = list(range(10, 40))
        result = hedges_g(g1, g2)
        assert abs(result["g"] - result["d"]) < 0.1


class TestDescriptiveStats:
    def test_basic(self):
        result = descriptive_stats([1, 2, 3, 4, 5])
        assert result["n"] == 5
        assert result["mean"] == 3.0
        assert result["min"] == 1
        assert result["max"] == 5

    def test_empty(self):
        result = descriptive_stats([])
        assert result["n"] == 0


class TestDetectTrend:
    def test_rising_trend(self):
        ts = ["2026-01-01"] * 10
        vals = list(range(10))
        result = detect_trend(ts, vals)
        assert result["trend_direction"] == "上升"

    def test_stable(self):
        ts = ["2026-01-01"] * 5
        vals = [5, 5, 5, 5, 5]
        result = detect_trend(ts, vals)
        assert result["trend_direction"] == "稳定"


class TestDetectAnomalies:
    def test_zscore_no_anomalies(self):
        result = detect_anomalies([5, 6, 5, 6, 5, 6], threshold=2.5)
        assert result["n_anomalies"] == 0

    def test_zscore_finds_outlier(self):
        result = detect_anomalies([5, 6, 5, 6, 5, 25], threshold=2.0)
        assert result["n_anomalies"] == 1

    def test_iqr_method(self):
        result = detect_anomalies([1, 2, 3, 4, 5, 50], method="iqr", threshold=1.5)
        assert result["n_anomalies"] >= 1


class TestVolatilityEvents:
    def test_detect_events(self):
        records = [
            {"anxiety_score": 2.0, "timestamp": "2026-01-01"},
            {"anxiety_score": 8.0, "timestamp": "2026-01-02"},
            {"anxiety_score": 2.5, "timestamp": "2026-01-03"},
        ]
        events = detect_high_volatility_events(records, threshold=3.0)
        assert len(events) >= 1


class TestCompareDistributions:
    def test_different_distributions(self):
        d1 = {"平静": 10, "开心": 5}
        d2 = {"平静": 5, "开心": 10, "悲伤": 5}
        result = compare_emotion_distributions(d1, d2)
        assert result["total_variation_distance"] > 0.1


class TestMCID:
    def test_half_sd(self):
        result = estimate_mcid([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert result["mcid"] > 0
        assert result["method"] == "half_sd"
