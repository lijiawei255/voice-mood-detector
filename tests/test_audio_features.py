# -*- coding: utf-8 -*-
"""声学特征提取模块 — 单元测试"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wave
import pytest
from audio_features import (
    extract_acoustic_features,
    compute_stress_index,
    compute_anxiety_index,
    compute_depression_tendency_index,
    compute_emotional_activation,
    compute_speech_stability,
    compute_psychological_indicators,
)


def _create_test_wav(filepath, duration_sec=3.0, sr=16000, amp=0.5, freq=200.0):
    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    data = (amp * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())
    return filepath


class TestExtractAcousticFeatures:
    def test_extract_valid_file(self, tmp_path):
        fp = str(tmp_path / "test.wav")
        _create_test_wav(fp, duration_sec=3.0)
        features = extract_acoustic_features(fp)
        assert features is not None
        assert 'rms_mean' in features
        assert 'rms_std' in features
        assert 'silence_ratio' in features
        assert 'short_term_energy' in features
        assert 'f0_mean' in features or 'voiced_ratio' in features
        assert 'mfcc_1_mean' in features
        assert 'hnr_mean' in features

    def test_missing_file_returns_none(self):
        features = extract_acoustic_features("/nonexistent.wav")
        assert features is None


class TestPsychologicalIndicators:
    def setup_method(self):
        self.mock_features = {
            'rms_mean': 0.1, 'rms_std': 0.03,
            'silence_ratio': 0.2, 'speech_rate': 4.0,
            'short_term_energy': 0.08,
            'f0_mean': 150, 'f0_std': 30,
            'jitter_local': 0.005,
            'shimmer_local': 0.02,
            'hnr_mean': 22,
        }

    def test_stress_index_range(self):
        si = compute_stress_index(self.mock_features)
        assert 0.0 <= si <= 1.0

    def test_anxiety_index_range(self):
        ai = compute_anxiety_index(self.mock_features)
        assert 0.0 <= ai <= 1.0

    def test_depression_tendency_range(self):
        di = compute_depression_tendency_index(self.mock_features)
        assert 0.0 <= di <= 1.0

    def test_emotional_activation_range(self):
        ea = compute_emotional_activation(self.mock_features)
        assert 0.0 <= ea <= 1.0

    def test_speech_stability_range(self):
        ss = compute_speech_stability(self.mock_features)
        assert 0.0 <= ss <= 1.0

    def test_high_stability_for_good_features(self):
        good = dict(self.mock_features)
        good.update({'jitter_local': 0.001, 'shimmer_local': 0.005, 'hnr_mean': 28})
        ss = compute_speech_stability(good)
        assert ss > 0.7

    def test_psychological_indicators_returns_all(self):
        result = compute_psychological_indicators(self.mock_features)
        assert 'stress_index' in result
        assert 'anxiety_index' in result
        assert 'depression_tendency_index' in result
        assert 'emotional_activation' in result
        assert 'speech_stability' in result

    def test_empty_features_handled(self):
        result = compute_psychological_indicators(None)
        assert result is not None
        assert result['stress_index'] is not None
