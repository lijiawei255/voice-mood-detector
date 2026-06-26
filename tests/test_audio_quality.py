# -*- coding: utf-8 -*-
"""
音频质量评估模块 — 单元测试

测试覆盖：
- trim_silence: 前后静音裁剪
- normalize_volume: 音量归一化
- detect_speech_segments: 语音段检测
- compute_audio_quality: 质量指标计算
- AudioQualityAnalyzer: 完整分析流程
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import wave
import tempfile
import pytest

from audio_quality import (
    AudioQualityAnalyzer,
    trim_silence,
    normalize_volume,
    simple_noise_reduce,
    detect_speech_segments,
    compute_audio_quality,
    AUDIO_QUALITY_THRESHOLDS,
    load_audio,
)


def _create_test_wav(filepath, duration_sec=3.0, sample_rate=16000, amplitude=0.5,
                     silence=False, clipping=False, freq=440.0):
    """Helper: create a test WAV file with specified properties."""
    n_samples = int(duration_sec * sample_rate)
    if silence:
        data = np.zeros(n_samples, dtype=np.int16)
    else:
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        data = (amplitude * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
        if clipping:
            data = np.clip(data, -20000, 20000)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())
    return filepath


class TestLoadAudio:
    def test_load_valid_wav(self, tmp_path):
        fp = str(tmp_path / "test.wav")
        _create_test_wav(fp, duration_sec=2.0)
        audio, sr = load_audio(fp)
        assert audio is not None
        assert len(audio) > 0
        assert sr == 16000

    def test_load_nonexistent(self):
        audio, sr = load_audio("/nonexistent/path.wav")
        assert audio is None


class TestTrimSilence:
    def test_trim_silence_removes_leading_trailing(self):
        sr = 16000
        silence = np.zeros(int(0.5 * sr), dtype=np.float32)
        tone = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr, endpoint=False)).astype(np.float32) * 0.5
        signal = np.concatenate([silence, tone, silence])
        trimmed = trim_silence(signal, sr)
        assert len(trimmed) < len(signal)
        assert len(trimmed) >= sr * 0.7

    def test_trim_silence_all_silence_returns_empty(self):
        sr = 16000
        signal = np.zeros(int(2.0 * sr), dtype=np.float32)
        trimmed = trim_silence(signal, sr)
        # All silence should return very short or empty
        assert len(trimmed) < sr * 0.5

    def test_trim_silence_all_tone_returns_similar(self):
        sr = 16000
        tone = np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, 2*sr, endpoint=False)).astype(np.float32) * 0.5
        trimmed = trim_silence(tone, sr)
        assert len(trimmed) >= sr * 1.5


class TestNormalizeVolume:
    def test_normalize_volume_scales_to_target(self):
        sr = 16000
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr, endpoint=False)).astype(np.float32) * 0.1
        normalized = normalize_volume(signal, target_rms=0.2)
        rms = np.sqrt(np.mean(normalized ** 2))
        assert abs(rms - 0.2) < 0.1

    def test_normalize_silence_returns_unchanged(self):
        signal = np.zeros(1000, dtype=np.float32)
        normalized = normalize_volume(signal)
        assert np.allclose(normalized, 0.0)


class TestDetectSpeechSegments:
    def test_detect_speech_in_mixed_signal(self):
        sr = 16000
        tone = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)).astype(np.float32) * 0.5
        silence = np.zeros(int(0.5 * sr), dtype=np.float32)  # 0.5s silence > min_silence_duration
        signal = np.concatenate([silence, tone, silence, tone, silence])
        segments = detect_speech_segments(signal, sr, energy_threshold=0.001)
        assert len(segments) >= 2

    def test_all_silence_no_segments(self):
        sr = 16000
        signal = np.zeros(int(2.0 * sr), dtype=np.float32)
        segments = detect_speech_segments(signal, sr)
        assert len(segments) == 0


class TestComputeAudioQuality:
    def test_good_quality_signal(self, tmp_path):
        fp = str(tmp_path / "good.wav")
        _create_test_wav(fp, duration_sec=5.0, amplitude=0.5)
        result = compute_audio_quality(fp)
        assert result is not None
        assert result["quality_score"] > 0.5
        assert result["speech_ratio"] > 0.5
        assert "duration" in result
        assert "rms_mean" in result
        assert "clipping_ratio" in result
        assert "noise_level" in result
        assert "quality_label" in result

    def test_silent_signal(self, tmp_path):
        fp = str(tmp_path / "silent.wav")
        _create_test_wav(fp, duration_sec=3.0, silence=True)
        result = compute_audio_quality(fp)
        assert result is not None
        # Silent signal should have low speech ratio or low quality
        assert result["speech_ratio"] < 0.5 or result["quality_score"] < 0.6

    def test_missing_file_returns_none(self):
        result = compute_audio_quality("/nonexistent/path.wav")
        assert result is None


class TestAudioQualityAnalyzer:
    def test_analyzer_valid_file(self, tmp_path):
        fp = str(tmp_path / "test.wav")
        _create_test_wav(fp, duration_sec=3.0, amplitude=0.5)
        analyzer = AudioQualityAnalyzer()
        result = analyzer.analyze(fp)
        assert result is not None
        assert "duration" in result
        assert "speech_duration" in result
        assert "speech_ratio" in result
        assert "rms_mean" in result
        assert "rms_std" in result
        assert "clipping_ratio" in result
        assert "noise_level" in result
        assert "quality_score" in result
        assert "quality_label" in result
        assert "issues" in result
        assert "recommendations" in result

    def test_analyzer_nonexistent_file(self):
        analyzer = AudioQualityAnalyzer()
        result = analyzer.analyze("/nonexistent.wav")
        assert result is None
