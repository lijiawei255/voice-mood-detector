# -*- coding: utf-8 -*-
"""
VAD维度估计模块 — 单元测试

测试覆盖：
- estimate_valence: 效价估计
- estimate_arousal: 唤醒度估计
- estimate_dominance: 掌控感估计
- compute_negative_load: 负性负荷
- compute_emotional_uncertainty: 情绪不确定性
- compute_vad_dimensions: 完整维度计算
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from vad_dimensions import (
    estimate_valence, estimate_arousal, estimate_dominance,
    compute_vad_dimensions, compute_negative_load,
    compute_emotional_uncertainty, VAD_ESTIMATION_NOTE
)


# Test fixtures
FIXTURE_POSITIVE = {
    "愤怒": 0.05, "厌恶": 0.02, "恐惧": 0.03,
    "开心": 0.60, "平静": 0.20, "其他": 0.02,
    "悲伤": 0.03, "惊讶": 0.05
}

FIXTURE_NEGATIVE = {
    "愤怒": 0.40, "厌恶": 0.05, "恐惧": 0.30,
    "开心": 0.02, "平静": 0.08, "其他": 0.02,
    "悲伤": 0.10, "惊讶": 0.03
}

FIXTURE_NEUTRAL = {
    "愤怒": 0.02, "厌恶": 0.02, "恐惧": 0.02,
    "开心": 0.10, "平静": 0.70, "其他": 0.05,
    "悲伤": 0.04, "惊讶": 0.05
}

FIXTURE_CONCENTRATED = {
    "愤怒": 0.0, "厌恶": 0.0, "恐惧": 0.0, "开心": 0.9,
    "平静": 0.05, "其他": 0.01, "悲伤": 0.02, "惊讶": 0.02
}

FIXTURE_DISPERSED = {
    "愤怒": 0.15, "厌恶": 0.10, "恐惧": 0.15, "开心": 0.15,
    "平静": 0.15, "其他": 0.10, "悲伤": 0.10, "惊讶": 0.10
}


class TestEstimateValence:
    def test_positive_emotion_gives_positive_valence(self):
        v = estimate_valence(FIXTURE_POSITIVE)
        assert v > 0, f"Expected positive valence, got {v}"

    def test_negative_emotion_gives_negative_valence(self):
        v = estimate_valence(FIXTURE_NEGATIVE)
        assert v < 0, f"Expected negative valence, got {v}"

    def test_neutral_gives_near_zero(self):
        v = estimate_valence(FIXTURE_NEUTRAL)
        assert -0.5 < v < 0.5

    def test_valence_range(self):
        v = estimate_valence(FIXTURE_POSITIVE)
        assert -1.0 <= v <= 1.0


class TestEstimateArousal:
    def test_high_energy_emotions_give_high_arousal(self):
        high_energy = {"愤怒": 0.3, "恐惧": 0.3, "开心": 0.2, "惊讶": 0.1,
                       "厌恶": 0.02, "平静": 0.03, "其他": 0.02, "悲伤": 0.03}
        a = estimate_arousal(high_energy)
        assert a > 0.4

    def test_low_energy_emotions_give_low_arousal(self):
        low_energy = {"悲伤": 0.5, "平静": 0.3, "其他": 0.05,
                      "愤怒": 0.03, "厌恶": 0.02, "恐惧": 0.02, "开心": 0.03, "惊讶": 0.05}
        a = estimate_arousal(low_energy)
        assert a < 0.5

    def test_arousal_range(self):
        a = estimate_arousal(FIXTURE_POSITIVE)
        assert 0.0 <= a <= 1.0


class TestEstimateDominance:
    def test_fear_and_sadness_lower_dominance(self):
        low_dom = {"愤怒": 0.05, "恐惧": 0.4, "悲伤": 0.3, "厌恶": 0.05,
                   "开心": 0.05, "平静": 0.1, "其他": 0.02, "惊讶": 0.03}
        d = estimate_dominance(low_dom)
        assert d < 0.6

    def test_happy_and_neutral_higher_dominance(self):
        high_dom = {"愤怒": 0.02, "恐惧": 0.02, "悲伤": 0.02, "厌恶": 0.02,
                    "开心": 0.4, "平静": 0.4, "其他": 0.05, "惊讶": 0.07}
        d = estimate_dominance(high_dom)
        assert d > 0.4

    def test_dominance_range(self):
        d = estimate_dominance(FIXTURE_POSITIVE)
        assert 0.0 <= d <= 1.0


class TestComputeVADDimensions:
    def test_returns_all_fields(self):
        result = compute_vad_dimensions(FIXTURE_POSITIVE)
        assert "valence_score" in result
        assert "arousal_score" in result
        assert "dominance_score" in result
        assert "negative_load" in result
        assert "emotional_uncertainty" in result
        assert "estimation_note" in result
        assert len(result["estimation_note"]) > 0

    def test_empty_probs_handles_gracefully(self):
        result = compute_vad_dimensions({})
        assert result["valence_score"] == 0.0
        assert result["arousal_score"] == 0.0
        assert result["dominance_score"] == 0.0
        assert result["negative_load"] == 0.0
        assert result["emotional_uncertainty"] == 0.0

    def test_none_probs_handles_gracefully(self):
        result = compute_vad_dimensions(None)
        assert result["valence_score"] == 0.0


class TestNegativeLoad:
    def test_negative_emotions_high_load(self):
        nl = compute_negative_load(FIXTURE_NEGATIVE)
        assert nl > 0.5

    def test_positive_emotions_low_load(self):
        nl = compute_negative_load(FIXTURE_POSITIVE)
        assert nl < 0.3


class TestEmotionalUncertainty:
    def test_concentrated_distribution_low_uncertainty(self):
        eu = compute_emotional_uncertainty(FIXTURE_CONCENTRATED)
        assert eu < 0.4, f"Expected low uncertainty, got {eu}"

    def test_dispersed_distribution_high_uncertainty(self):
        eu = compute_emotional_uncertainty(FIXTURE_DISPERSED)
        assert eu > 0.5, f"Expected high uncertainty, got {eu}"

    def test_uncertainty_range(self):
        eu = compute_emotional_uncertainty(FIXTURE_POSITIVE)
        assert 0.0 <= eu <= 1.0


def test_vad_estimation_note_is_meaningful():
    """确保标注说明包含必要信息"""
    assert len(VAD_ESTIMATION_NOTE) > 20
    # Should mention derivation / estimation / 推导
    note_lower = VAD_ESTIMATION_NOTE.lower()
    assert any(word in note_lower for word in ['推导', '估计', 'derived', 'estimated', 'inferred'])
