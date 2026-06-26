# Voice Mood Detect — 科研级升级实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade voice emotion recognition system into a standardized, explainable, reproducible voice psychological state assessment platform (P0→P1→P2).

**Architecture:** Modular additions to existing codebase — 7 new modules, 4 existing modules modified. Data flow: Audio → Quality Gate → Acoustic Features → Emotion Model(s) → VAD Estimation → Compound Scoring → Stability → Baseline Comparison → Reliability → Structured Report.

**Tech Stack:** Python 3.x (Anaconda `audio` env), PyQt5, emotion2vec+ via FunASR, librosa, praat-parselmouth (P1), matplotlib, scipy, numpy.

**Conventions:** TDD (test-first), atomic commits at task boundaries, Chinese UI labels, CPU-only inference.

---

## File Structure Map

| File | Action | Responsibility |
|------|--------|---------------|
| `audio_quality.py` | CREATE | Audio pre-processing, VAD, quality metrics |
| `vad_dimensions.py` | CREATE | Valence-Arousal-Dominance estimation from discrete emotions |
| `audio_features.py` | CREATE (P1) | Acoustic feature extraction (F0, jitter, shimmer, HNR, MFCC) |
| `reliability.py` | CREATE (P1) | ICC, test-retest, dual-model agreement, multi-sample evaluation |
| `export_manager.py` | CREATE (P1) | CSV/JSON research data export |
| `baseline.py` | CREATE (P2) | Personal baseline modeling and deviation tracking |
| `statistics.py` | CREATE (P2) | Effect size, statistical tests, anomaly detection |
| `gui_widgets/__init__.py` | CREATE | Package init |
| `gui_widgets/result_cards.py` | CREATE (P0) | Structured assessment report card widgets |
| `gui_widgets/research_panel.py` | CREATE (P1) | Research mode UI panel |
| `gui_widgets/stats_panel.py` | CREATE (P1) | History statistics visualization panel |
| `gui_widgets/baseline_panel.py` | CREATE (P2) | Baseline management UI |
| `emotion_recognizer.py` | MODIFY | Sub-scores, VAD output, raw output, dual-model, compound scoring |
| `history_manager.py` | MODIFY | Extended schema, research mode, statistics APIs, export |
| `recorder.py` | MODIFY | Real-time quality monitoring |
| `gui.py` | MODIFY | Integrate new widgets, research mode, quality feedback |
| `main.py` | MODIFY | Version constants |
| `relaxation_tips.py` | REVIEW | Ensure compatibility (no structural changes needed) |

---

## Phase 1: P0 Core Improvements

### Task 1.1: Create `audio_quality.py` module

**Files:**
- Create: `audio_quality.py`
- Test: `test_audio_quality.py`

- [ ] **Step 1: Write failing tests**

```python
# test_audio_quality.py
import os
import sys
import numpy as np
import wave
import tempfile
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_quality import (
    AudioQualityAnalyzer,
    trim_silence,
    normalize_volume,
    simple_noise_reduce,
    detect_speech_segments,
    compute_audio_quality,
    AUDIO_QUALITY_THRESHOLDS
)


def _create_test_wav(filepath, duration_sec=3.0, sample_rate=16000, amplitude=0.5, silence=False):
    """Helper: create a test WAV file."""
    n_samples = int(duration_sec * sample_rate)
    if silence:
        data = np.zeros(n_samples, dtype=np.int16)
    else:
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        data = (amplitude * 32767 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())
    return filepath


class TestTrimSilence:
    def test_trim_silence_removes_leading_trailing(self):
        """Silence at start and end should be removed."""
        sr = 16000
        # 0.5s silence + 1s tone + 0.5s silence
        silence = np.zeros(int(0.5 * sr), dtype=np.float32)
        tone = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr, endpoint=False)).astype(np.float32) * 0.5
        signal = np.concatenate([silence, tone, silence])
        trimmed = trim_silence(signal, sr)
        # Trimmed should be shorter than original
        assert len(trimmed) < len(signal)
        # Should contain the tone portion
        assert len(trimmed) >= sr * 0.8

    def test_trim_silence_all_silence_returns_short(self):
        """All-silence signal should return very short array."""
        sr = 16000
        signal = np.zeros(int(2.0 * sr), dtype=np.float32)
        trimmed = trim_silence(signal, sr)
        assert len(trimmed) < sr * 0.5


class TestNormalizeVolume:
    def test_normalize_volume_scales_to_target(self):
        """Signal should be scaled to target RMS."""
        sr = 16000
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr, endpoint=False)).astype(np.float32) * 0.1
        normalized = normalize_volume(signal, target_rms=0.2)
        rms = np.sqrt(np.mean(normalized ** 2))
        assert abs(rms - 0.2) < 0.05

    def test_normalize_silence_returns_unchanged(self):
        """Silence should not be amplified."""
        signal = np.zeros(1000, dtype=np.float32)
        normalized = normalize_volume(signal)
        assert np.all(normalized == 0.0)


class TestDetectSpeechSegments:
    def test_detect_speech_in_mixed_signal(self):
        """Should find speech segments in signal with gaps."""
        sr = 16000
        tone = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)).astype(np.float32) * 0.5
        silence = np.zeros(int(0.3 * sr), dtype=np.float32)
        signal = np.concatenate([silence, tone, silence, tone, silence])
        segments = detect_speech_segments(signal, sr)
        assert len(segments) >= 2


class TestComputeAudioQuality:
    def test_good_quality_signal(self, tmp_path):
        """Clean tone should get high quality score."""
        fp = str(tmp_path / "good.wav")
        _create_test_wav(fp, duration_sec=5.0, amplitude=0.5)
        result = compute_audio_quality(fp)
        assert result["quality_score"] > 0.5
        assert result["speech_ratio"] > 0.5
        assert "duration" in result
        assert "rms_mean" in result
        assert "clipping_ratio" in result
        assert "noise_level" in result

    def test_silent_signal_low_quality(self, tmp_path):
        """Silent recording should get low quality score."""
        fp = str(tmp_path / "silent.wav")
        _create_test_wav(fp, duration_sec=3.0, silence=True)
        result = compute_audio_quality(fp)
        assert result["speech_ratio"] < 0.3 or result["quality_score"] < 0.5

    def test_missing_file_returns_none(self):
        """Non-existent file should return None."""
        result = compute_audio_quality("/nonexistent/path.wav")
        assert result is None


class TestAudioQualityAnalyzer:
    def test_analyzer_full_pipeline(self, tmp_path):
        """Full analyzer pipeline should return all quality metrics."""
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

    def test_analyzer_rejects_bad_input(self):
        """Non-existent file should return None."""
        analyzer = AudioQualityAnalyzer()
        result = analyzer.analyze("/nonexistent.wav")
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd c:/Work/Voice_Mood_Detect && conda run -n audio python -m pytest test_audio_quality.py -v 2>&1 | tail -5
```
Expected: FAIL (module not found)

- [ ] **Step 3: Write `audio_quality.py` implementation**

```python
# -*- coding: utf-8 -*-
"""
语音情绪识别系统 - 音频质量评估模块

本模块负责音频预处理和质量评估，包括：
1. 语音活动检测 (VAD)
2. 自动裁剪前后静音
3. 音量归一化
4. 简单降噪（谱减法）
5. 音频质量指标计算

质量指标：
- duration: 总时长
- speech_duration: 有效语音时长
- speech_ratio: 有效语音比例
- rms_mean / rms_std: 音量均值/标准差
- clipping_ratio: 爆音比例
- noise_level: 估计噪声水平
- quality_score: 综合质量分数 (0.0-1.0)

科研使用建议：
- quality_score >= 0.6: 适合分析
- quality_score 0.4-0.6: 勉强可用
- quality_score < 0.4: 建议重新录制

作者：Jiawei Li
许可证：GPL v3
"""

import os
import numpy as np
import logging
import wave

logger = logging.getLogger(__name__)

# 音频质量门控阈值
AUDIO_QUALITY_THRESHOLDS = {
    "min_duration": 1.0,           # 最小时长（秒）
    "max_duration": 60.0,          # 最大时长（秒）
    "min_speech_ratio": 0.4,       # 最小有效语音比例
    "max_clipping_ratio": 0.05,    # 最大爆音比例
    "max_noise_level": 0.3,        # 最大噪声水平
    "min_rms": 0.005,              # 最小RMS音量
    "research_min_duration": 3.0,  # 科研模式最小时长
    "research_max_duration": 30.0, # 科研模式最大时长
}


def load_audio(filepath, target_sr=16000):
    """
    加载音频文件并重采样到目标采样率

    参数：
        filepath (str): 音频文件路径
        target_sr (int): 目标采样率，默认 16000

    返回值：
        tuple: (audio_data, sample_rate) 或 (None, None) 失败时
    """
    try:
        import librosa
        audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
        return audio.astype(np.float32), sr
    except ImportError:
        # 回退：使用 wave + scipy 或 numpy 手动加载
        try:
            import scipy.io.wavfile as wavfile
            sr, data = wavfile.read(filepath)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            else:
                data = data.astype(np.float32)
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            # 简单重采样（线性插值）
            if sr != target_sr:
                import scipy.signal as signal
                data = signal.resample(data, int(len(data) * target_sr / sr))
                sr = target_sr
            return data.astype(np.float32), sr
        except Exception:
            pass
        # 手动读取 WAV
        try:
            with wave.open(filepath, 'rb') as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
                data = data.astype(np.float32) / 32768.0
            if sr != target_sr:
                from scipy import signal as scipy_signal
                data = scipy_signal.resample(data, int(len(data) * target_sr / sr))
            return data.astype(np.float32), target_sr
        except Exception as e:
            logger.error(f"加载音频失败: {e}")
            return None, None


def trim_silence(audio, sr, top_db=30, frame_length=2048, hop_length=512):
    """
    裁剪前后静音段

    参数：
        audio (np.ndarray): 音频数据
        sr (int): 采样率
        top_db (int): 低于此dB值视为静音
        frame_length (int): 帧长度
        hop_length (int): 跳跃长度

    返回值：
        np.ndarray: 裁剪后的音频
    """
    try:
        import librosa
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db,
                                          frame_length=frame_length,
                                          hop_length=hop_length)
        return trimmed
    except ImportError:
        # 简单实现：基于RMS能量阈值
        if len(audio) == 0:
            return audio
        frame_len = frame_length
        hop = hop_length
        n_frames = max(1, (len(audio) - frame_len) // hop + 1)
        rms = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop
            end = min(start + frame_len, len(audio))
            chunk = audio[start:end]
            rms[i] = np.sqrt(np.mean(chunk ** 2) + 1e-10)

        rms_db = 20 * np.log10(rms + 1e-10)
        threshold_db = -top_db
        non_silent = rms_db > threshold_db

        if not np.any(non_silent):
            return np.array([], dtype=np.float32)

        first = np.argmax(non_silent)
        last = len(non_silent) - np.argmax(non_silent[::-1]) - 1
        start_sample = max(0, first * hop)
        end_sample = min(len(audio), (last + 1) * hop + frame_len)
        return audio[start_sample:end_sample]


def normalize_volume(audio, target_rms=0.15):
    """
    音量归一化到目标RMS水平

    参数：
        audio (np.ndarray): 音频数据
        target_rms (float): 目标RMS值

    返回值：
        np.ndarray: 归一化后的音频
    """
    rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    if rms < 1e-8:
        return audio
    gain = target_rms / rms
    # 限制增益不超过 5 倍（防止过度放大噪声）
    gain = min(gain, 5.0)
    return audio * gain


def simple_noise_reduce(audio, sr, noise_reduce_factor=0.02):
    """
    简单谱减法降噪

    参数：
        audio (np.ndarray): 音频数据
        sr (int): 采样率
        noise_reduce_factor (float): 降噪强度

    返回值：
        np.ndarray: 降噪后的音频
    """
    try:
        import librosa
        # 使用前 0.5 秒估计噪声
        noise_sample = min(int(0.5 * sr), len(audio) // 4)
        if noise_sample < sr // 10:
            return audio
        noise_profile = audio[:noise_sample]
        noise_power = np.mean(noise_profile ** 2)

        # STFT
        D = librosa.stft(audio, n_fft=2048, hop_length=512)
        mag, phase = librosa.magphase(D)

        # 谱减法
        noise_mag = np.sqrt(noise_power) * noise_reduce_factor
        mag_reduced = np.maximum(mag - noise_mag, 0)

        # 逆STFT
        D_reduced = mag_reduced * phase
        y_reduced = librosa.istft(D_reduced, hop_length=512, length=len(audio))
        return y_reduced.astype(np.float32)
    except ImportError:
        return audio


def detect_speech_segments(audio, sr, frame_length=2048, hop_length=512,
                           energy_threshold=0.01, min_silence_duration=0.3):
    """
    检测语音活动段

    参数：
        audio (np.ndarray): 音频数据
        sr (int): 采样率
        energy_threshold (float): 能量阈值
        min_silence_duration (float): 最小静音间隔（秒）

    返回值：
        list: [(start_time, end_time), ...] 语音段列表
    """
    if len(audio) == 0:
        return []

    frame_len = frame_length
    hop = hop_length
    n_frames = max(1, (len(audio) - frame_len) // hop + 1)

    energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        end = min(start + frame_len, len(audio))
        chunk = audio[start:end]
        energy[i] = np.sqrt(np.mean(chunk ** 2) + 1e-10)

    is_speech = energy > energy_threshold

    # 合并相邻语音段
    min_silence_frames = int(min_silence_duration * sr / hop)
    segments = []
    in_speech = False
    speech_start = 0

    for i in range(len(is_speech)):
        if is_speech[i] and not in_speech:
            speech_start = i * hop / sr
            in_speech = True
        elif not is_speech[i] and in_speech:
            # 检查静音是否持续足够长
            silence_count = 0
            for j in range(i, min(len(is_speech), i + min_silence_frames)):
                if not is_speech[j]:
                    silence_count += 1
                else:
                    break
            if silence_count >= min_silence_frames:
                speech_end = (i - 1) * hop / sr
                segments.append((speech_start, speech_end))
                in_speech = False

    if in_speech:
        speech_end = len(audio) / sr
        segments.append((speech_start, speech_end))

    return segments


def compute_audio_quality(filepath):
    """
    计算音频质量指标

    参数：
        filepath (str): 音频文件路径

    返回值：
        dict or None: 质量指标字典
    """
    if not filepath or not os.path.exists(filepath):
        return None

    try:
        audio, sr = load_audio(filepath)
        if audio is None or len(audio) == 0:
            return None

        duration = len(audio) / sr

        # RMS 统计
        rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
        frame_len = 2048
        hop = 512
        n_frames = max(1, (len(audio) - frame_len) // hop + 1)
        rms_per_frame = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop
            end = min(start + frame_len, len(audio))
            chunk = audio[start:end]
            rms_per_frame[i] = np.sqrt(np.mean(chunk ** 2) + 1e-10)
        rms_std = float(np.std(rms_per_frame))

        # 爆音检测 (clipping ratio)
        clipping_ratio = float(np.mean(np.abs(audio) > 0.95))

        # 语音活动检测
        segments = detect_speech_segments(audio, sr, energy_threshold=max(0.005, rms * 0.3))
        speech_duration = sum(end - start for start, end in segments)
        speech_ratio = speech_duration / duration if duration > 0 else 0.0

        # 噪声水平估计（非语音段的平均能量）
        if speech_ratio > 0.1 and speech_ratio < 0.9:
            # 有足够非语音段来估计噪声
            speech_mask = np.zeros(len(audio), dtype=bool)
            for start, end in segments:
                speech_mask[int(start * sr):int(end * sr)] = True
            non_speech = audio[~speech_mask]
            if len(non_speech) > 0:
                noise_rms = float(np.sqrt(np.mean(non_speech ** 2) + 1e-10))
            else:
                noise_rms = 0.0
        else:
            # 全是语音或全是静音
            noise_rms = 0.0 if speech_ratio > 0.9 else float(rms)

        noise_level = noise_rms / max(rms, 1e-8)
        noise_level = min(1.0, noise_level)

        # 综合质量分数
        # 考虑：语音比例、爆音比例、噪声水平、音量适中度
        speech_score = min(1.0, speech_ratio / 0.6)  # 语音比例越高越好
        clip_score = max(0.0, 1.0 - clipping_ratio / AUDIO_QUALITY_THRESHOLDS["max_clipping_ratio"])
        noise_score = max(0.0, 1.0 - noise_level / AUDIO_QUALITY_THRESHOLDS["max_noise_level"])
        # 音量适中度（在0.01-0.3之间最好）
        rms_ideal = 0.1
        rms_score = max(0.0, 1.0 - abs(rms - rms_ideal) / rms_ideal)

        quality_score = (
            speech_score * 0.35 +
            clip_score * 0.25 +
            noise_score * 0.25 +
            rms_score * 0.15
        )
        quality_score = round(min(1.0, max(0.0, quality_score)), 4)

        return {
            "duration": round(duration, 2),
            "speech_duration": round(speech_duration, 2),
            "speech_ratio": round(speech_ratio, 4),
            "rms_mean": round(float(rms), 6),
            "rms_std": round(rms_std, 6),
            "clipping_ratio": round(clipping_ratio, 4),
            "noise_level": round(noise_level, 4),
            "quality_score": quality_score,
            "quality_label": _quality_label(quality_score),
            "sample_rate": sr
        }
    except Exception as e:
        logger.error(f"计算音频质量失败: {e}")
        return None


def _quality_label(score):
    """将质量分数转换为标签"""
    if score >= 0.8:
        return "优秀"
    elif score >= 0.6:
        return "良好"
    elif score >= 0.4:
        return "一般"
    elif score >= 0.2:
        return "较差"
    else:
        return "很差"


class AudioQualityAnalyzer:
    """
    音频质量分析器

    封装完整的音频预处理和质量评估流程：
    1. 加载音频
    2. 裁剪静音
    3. 降噪
    4. 音量归一化
    5. 质量评估
    """

    def __init__(self, target_sr=16000):
        self.target_sr = target_sr

    def analyze(self, filepath):
        """
        对音频文件进行完整的质量分析

        参数：
            filepath (str): 音频文件路径

        返回值：
            dict or None: 包含完整质量信息的字典
        """
        if not filepath or not os.path.exists(filepath):
            logger.warning(f"音频文件不存在: {filepath}")
            return None

        try:
            audio, sr = load_audio(filepath, self.target_sr)
            if audio is None or len(audio) == 0:
                return None

            # 保存原始时长
            original_duration = len(audio) / sr

            # 裁剪静音
            trimmed = trim_silence(audio, sr)

            # 降噪
            denoised = simple_noise_reduce(trimmed, sr)

            # 计算质量指标
            quality = compute_audio_quality(filepath)
            if quality is None:
                return None

            # 重新计算裁剪后的指标
            trimmed_duration = len(trimmed) / sr

            # 识别质量问题
            issues = []
            recommendations = []

            thresholds = AUDIO_QUALITY_THRESHOLDS
            if original_duration < thresholds["min_duration"]:
                issues.append("录音时长过短")
                recommendations.append(f"建议录音至少 {thresholds['min_duration']} 秒")
            if quality["speech_ratio"] < thresholds["min_speech_ratio"]:
                issues.append("有效语音比例过低")
                recommendations.append("请靠近麦克风并清晰地说话")
            if quality["clipping_ratio"] > thresholds["max_clipping_ratio"]:
                issues.append("检测到爆音")
                recommendations.append("请适当远离麦克风或降低音量")
            if quality["noise_level"] > thresholds["max_noise_level"]:
                issues.append("环境噪声较高")
                recommendations.append("建议在安静的环境中录音")
            if quality["rms_mean"] < thresholds["min_rms"]:
                issues.append("音量过低")
                recommendations.append("请靠近麦克风或提高说话音量")

            quality["issues"] = issues
            quality["recommendations"] = recommendations
            quality["original_duration"] = round(original_duration, 2)
            quality["trimmed_duration"] = round(trimmed_duration, 2)

            return quality

        except Exception as e:
            logger.error(f"音频质量分析失败: {e}")
            return None

    def preprocess(self, filepath, output_path=None, trim_silence_flag=True,
                   normalize_volume_flag=True, reduce_noise_flag=True):
        """
        预处理音频文件

        参数：
            filepath (str): 输入文件路径
            output_path (str): 输出文件路径（可选）
            trim_silence_flag (bool): 是否裁剪静音
            normalize_volume_flag (bool): 是否归一化音量
            reduce_noise_flag (bool): 是否降噪

        返回值：
            tuple: (processed_audio, sample_rate) 或 (None, None)
        """
        audio, sr = load_audio(filepath, self.target_sr)
        if audio is None:
            return None, None

        if trim_silence_flag:
            audio = trim_silence(audio, sr)
        if reduce_noise_flag:
            audio = simple_noise_reduce(audio, sr)
        if normalize_volume_flag:
            audio = normalize_volume(audio)

        if output_path and audio is not None:
            try:
                import soundfile as sf
                sf.write(output_path, audio, sr)
            except ImportError:
                import scipy.io.wavfile as wavfile
                wavfile.write(output_path, sr, (audio * 32767).astype(np.int16))

        return audio, sr
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n audio python -m pytest test_audio_quality.py -v
```
Expected: All tests PASS

- [ ] **Step 5: Run existing tests to ensure no regression**

```bash
conda run -n audio python -m pytest test_emotion.py test_paths.py test_recording_flow.py -v
```

---

### Task 1.2: Create `vad_dimensions.py` module

**Files:**
- Create: `vad_dimensions.py`
- Test: `test_vad_dimensions.py`

- [ ] **Step 1: Write failing tests**

```python
# test_vad_dimensions.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pytest
from vad_dimensions import (
    estimate_valence, estimate_arousal, estimate_dominance,
    compute_vad_dimensions, compute_negative_load,
    compute_emotional_uncertainty, VAD_ESTIMATION_NOTE
)


FIXTURE_PROBS = {
    "愤怒": 0.05, "厌恶": 0.02, "恐惧": 0.03,
    "开心": 0.60, "平静": 0.20, "其他": 0.02,
    "悲伤": 0.03, "惊讶": 0.05
}

FIXTURE_NEGATIVE = {
    "愤怒": 0.40, "厌恶": 0.05, "恐惧": 0.30,
    "开心": 0.02, "平静": 0.08, "其他": 0.02,
    "悲伤": 0.10, "惊讶": 0.03
}


class TestEstimateValence:
    def test_positive_emotion_gives_positive_valence(self):
        v = estimate_valence(FIXTURE_PROBS)
        assert v > 0, f"Expected positive valence, got {v}"

    def test_negative_emotion_gives_negative_valence(self):
        v = estimate_valence(FIXTURE_NEGATIVE)
        assert v < 0, f"Expected negative valence, got {v}"

    def test_valence_range(self):
        v = estimate_valence(FIXTURE_PROBS)
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
        a = estimate_arousal(FIXTURE_PROBS)
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
        d = estimate_dominance(FIXTURE_PROBS)
        assert 0.0 <= d <= 1.0


class TestComputeVADDimensions:
    def test_returns_all_fields(self):
        result = compute_vad_dimensions(FIXTURE_PROBS)
        assert "valence_score" in result
        assert "arousal_score" in result
        assert "dominance_score" in result
        assert "negative_load" in result
        assert "emotional_uncertainty" in result
        assert "estimation_note" in result

    def test_empty_probs_handles_gracefully(self):
        result = compute_vad_dimensions({})
        assert result["valence_score"] == 0.0
        assert result["arousal_score"] == 0.0


class TestNegativeLoad:
    def test_negative_emotions_high_load(self):
        nl = compute_negative_load(FIXTURE_NEGATIVE)
        assert nl > 0.5

    def test_positive_emotions_low_load(self):
        nl = compute_negative_load(FIXTURE_PROBS)
        assert nl < 0.3


class TestEmotionalUncertainty:
    def test_concentrated_distribution_low_uncertainty(self):
        conc = {"愤怒": 0.0, "厌恶": 0.0, "恐惧": 0.0, "开心": 0.9,
                "平静": 0.05, "其他": 0.01, "悲伤": 0.02, "惊讶": 0.02}
        eu = compute_emotional_uncertainty(conc)
        assert eu < 0.4

    def test_dispersed_distribution_high_uncertainty(self):
        dispersed = {"愤怒": 0.15, "厌恶": 0.10, "恐惧": 0.15, "开心": 0.15,
                     "平静": 0.15, "其他": 0.10, "悲伤": 0.10, "惊讶": 0.10}
        eu = compute_emotional_uncertainty(dispersed)
        assert eu > 0.5


def test_vad_estimation_note_exists():
    """确保标注说明存在"""
    assert len(VAD_ESTIMATION_NOTE) > 0
    assert "推导" in VAD_ESTIMATION_NOTE or "估计" in VAD_ESTIMATION_NOTE
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n audio python -m pytest test_vad_dimensions.py -v 2>&1 | tail -5
```

- [ ] **Step 3: Write `vad_dimensions.py` implementation**

```python
# -*- coding: utf-8 -*-
"""
语音情绪识别系统 - 效价-唤醒度-掌控感维度估计模块

本模块从离散情绪概率分布映射到连续的情感维度指标。
⚠️ 重要声明：emotion2vec+ 原始模型并非在 VAD 维度标注数据上训练，
   这些指标是从离散情绪概率映射的二次推断估计值，不作为独立科研变量的唯一来源。
   建议 P2 阶段引入专门的维度情绪回归模型。

维度说明：
- Valence (效价): 情绪正负性，-1.0 ~ 1.0
- Arousal (唤醒度): 情绪激活水平，0.0 ~ 1.0
- Dominance (掌控感): 对当前状态的主观掌控程度，0.0 ~ 1.0
- Negative Load (负性负荷): 负面情绪整体强度，0.0 ~ 1.0
- Emotional Uncertainty (情绪不确定性): 基于概率熵，0.0 ~ 1.0

映射依据：
- 效价：正面情绪（开心）= +，负面情绪（愤怒/恐惧/悲伤/厌恶）= -
- 唤醒度：高唤醒（愤怒/恐惧/开心/惊讶）> 低唤醒（悲伤/平静/厌恶）
- 掌控感：高掌控（开心/平静/愤怒）> 低掌控（恐惧/悲伤/惊讶）

作者：Jiawei Li
许可证：GPL v3
"""

import numpy as np
import math

# 标注说明（所有VAD输出中必须包含此字段）
VAD_ESTIMATION_NOTE = (
    "⚠️ 从离散情绪概率推导的估计值（非直接连续维度测量），"
    "不作为独立科研变量的唯一来源。建议结合标准化心理量表使用。"
)

# 效价映射权重（基于情绪效价维度理论）
# 正值 = 正面情绪，负值 = 负面情绪
VALENCE_WEIGHTS = {
    "愤怒": -0.8,
    "厌恶": -0.7,
    "恐惧": -0.9,
    "开心": 0.9,
    "平静": 0.3,
    "悲伤": -0.8,
    "惊讶": 0.0,  # 惊讶可为正可为负，取中性
    "其他": 0.0
}

# 唤醒度映射权重（基于情绪激活水平理论）
# 高值 = 高激活，低值 = 低激活
AROUSAL_WEIGHTS = {
    "愤怒": 0.9,
    "厌恶": 0.4,
    "恐惧": 0.95,
    "开心": 0.8,
    "平静": 0.1,
    "悲伤": 0.2,
    "惊讶": 0.85,
    "其他": 0.3
}

# 掌控感映射权重（基于情绪控制维度理论）
# 高值 = 高掌控感
DOMINANCE_WEIGHTS = {
    "愤怒": 0.7,   # 愤怒有一定掌控感但偏对抗
    "厌恶": 0.5,
    "恐惧": 0.1,   # 恐惧时掌控感最低
    "开心": 0.9,   # 开心时掌控感最高
    "平静": 0.8,   # 平静也有高掌控感
    "悲伤": 0.2,   # 悲伤时掌控感低
    "惊讶": 0.3,   # 惊讶时掌控感较低
    "其他": 0.5
}


def estimate_valence(probs_dict):
    """
    从离散情绪概率估计效价（Valence）

    参数：
        probs_dict (dict): 8种情绪的概率分布

    返回值：
        float: 效价值，范围 -1.0 ~ 1.0
    """
    if not probs_dict:
        return 0.0
    valence = 0.0
    total_weight = 0.0
    for emotion, prob in probs_dict.items():
        try:
            p = float(prob)
            w = VALENCE_WEIGHTS.get(emotion, 0.0)
            valence += p * w
            total_weight += p
        except (TypeError, ValueError):
            continue
    if total_weight > 0.001:
        valence /= total_weight
    return round(max(-1.0, min(1.0, valence)), 4)


def estimate_arousal(probs_dict):
    """
    从离散情绪概率估计唤醒度（Arousal）

    参数：
        probs_dict (dict): 8种情绪的概率分布

    返回值：
        float: 唤醒度值，范围 0.0 ~ 1.0
    """
    if not probs_dict:
        return 0.0
    arousal = 0.0
    total_weight = 0.0
    for emotion, prob in probs_dict.items():
        try:
            p = float(prob)
            w = AROUSAL_WEIGHTS.get(emotion, 0.5)
            arousal += p * w
            total_weight += p
        except (TypeError, ValueError):
            continue
    if total_weight > 0.001:
        arousal /= total_weight
    return round(max(0.0, min(1.0, arousal)), 4)


def estimate_dominance(probs_dict):
    """
    从离散情绪概率估计掌控感（Dominance）

    参数：
        probs_dict (dict): 8种情绪的概率分布

    返回值：
        float: 掌控感值，范围 0.0 ~ 1.0
    """
    if not probs_dict:
        return 0.0
    dominance = 0.0
    total_weight = 0.0
    for emotion, prob in probs_dict.items():
        try:
            p = float(prob)
            w = DOMINANCE_WEIGHTS.get(emotion, 0.5)
            dominance += p * w
            total_weight += p
        except (TypeError, ValueError):
            continue
    if total_weight > 0.001:
        dominance /= total_weight
    return round(max(0.0, min(1.0, dominance)), 4)


def compute_negative_load(probs_dict):
    """
    计算负性负荷（Negative Load）

    综合负面情绪（愤怒、恐惧、悲伤、厌恶）的强度。

    参数：
        probs_dict (dict): 8种情绪的概率分布

    返回值：
        float: 负性负荷，范围 0.0 ~ 1.0
    """
    negative_map = {"愤怒": 0.9, "恐惧": 1.0, "悲伤": 0.85, "厌恶": 0.75}
    load = 0.0
    for emotion, weight in negative_map.items():
        try:
            p = float(probs_dict.get(emotion, 0.0))
            load += p * weight
        except (TypeError, ValueError):
            continue
    return round(max(0.0, min(1.0, load)), 4)


def compute_emotional_uncertainty(probs_dict):
    """
    计算情绪不确定性（基于 Shannon 熵归一化）

    高熵表示情绪分布分散，不确定性高；
    低熵表示情绪集中，识别结果更确定。

    参数：
        probs_dict (dict): 8种情绪的概率分布

    返回值：
        float: 情绪不确定性，范围 0.0 ~ 1.0
    """
    if not probs_dict:
        return 0.0
    entropy = 0.0
    for prob in probs_dict.values():
        try:
            p = float(prob)
            if p > 0.001:
                entropy -= p * math.log2(p)
        except (TypeError, ValueError):
            continue
    n = max(1, len(probs_dict))
    max_entropy = math.log2(n)
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0
    return round(max(0.0, min(1.0, normalized)), 4)


def compute_vad_dimensions(probs_dict):
    """
    从情绪概率分布计算完整的 VAD 维度指标

    参数：
        probs_dict (dict): 8种情绪的概率分布（完整的8类，包含"其他"）

    返回值：
        dict: 包含所有维度指标的字典
    """
    if not isinstance(probs_dict, dict):
        probs_dict = {}

    return {
        "valence_score": estimate_valence(probs_dict),
        "arousal_score": estimate_arousal(probs_dict),
        "dominance_score": estimate_dominance(probs_dict),
        "negative_load": compute_negative_load(probs_dict),
        "emotional_uncertainty": compute_emotional_uncertainty(probs_dict),
        "estimation_note": VAD_ESTIMATION_NOTE
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n audio python -m pytest test_vad_dimensions.py -v
```

---

### Task 1.3: Modify `emotion_recognizer.py` — Stability sub-scores, VAD output, raw output

**Files:**
- Modify: `emotion_recognizer.py` (update `_calculate_stability_score` and `predict` methods)
- Test: `test_emotion_algorithm.py` (update existing, add new assertions)

- [ ] **Step 1: Update `_calculate_stability_score` to return sub-scores**

In `emotion_recognizer.py`, modify `_calculate_stability_score` to return dict instead of float:

```python
def _calculate_stability_score(self, probs_dict):
    """
    计算情绪稳定度分数（多因子综合评分）及分项

    返回：
        dict: {
            "negative_weight_score": float,
            "entropy_score": float,
            "extremity_score": float,
            "stability_score": float,
            "stability_level": str,
            "stability_color": str,
            "factor_weights_source": str
        }
    """
    import math
    if not isinstance(probs_dict, dict):
        return {
            "negative_weight_score": 5.0,
            "entropy_score": 5.0,
            "extremity_score": 5.0,
            "stability_score": 5.0,
            "stability_level": "未知",
            "stability_color": "#95A5A6",
            "factor_weights_source": "专家设定（40%/30%/30%），基于情绪维度理论"
        }

    # --- 因子 1: 负面情绪加权分数 (0-10) ---
    negative_score = 0.0
    for emotion, prob in probs_dict.items():
        try:
            prob_f = float(prob)
            weight = float(EMOTION_WEIGHTS.get(emotion, 0.0))
            if 0 <= prob_f <= 1:
                negative_score += prob_f * weight * 10
        except (TypeError, ValueError):
            continue
    negative_score = round(min(10.0, max(0.0, negative_score)), 2)

    # --- 因子 2: 情绪分散度/熵 (0-10) ---
    entropy = 0.0
    probs_list = []
    for prob in probs_dict.values():
        try:
            p = float(prob)
            if p > 0.001:
                probs_list.append(p)
                entropy -= p * math.log2(p)
        except (TypeError, ValueError):
            continue
    max_entropy = math.log2(len(probs_dict)) if len(probs_dict) > 1 else 1.0
    entropy_normalized = (entropy / max_entropy) * 10.0 if max_entropy > 0 else 0.0
    entropy_score = round(min(10.0, max(0.0, entropy_normalized)), 2)

    # --- 因子 3: 情绪极端度 (0-10) ---
    extremity_score = 0.0
    positive_total = 0.0
    for emotion, prob in probs_dict.items():
        try:
            prob_f = float(prob)
            if emotion in NEGATIVE_EMOTIONS:
                if prob_f > 0.60:
                    extremity_score = max(extremity_score, prob_f * 10)
                elif prob_f > 0.40:
                    extremity_score = max(extremity_score, prob_f * 7)
            if emotion in POSITIVE_EMOTIONS:
                positive_total += prob_f
        except (TypeError, ValueError):
            continue
    if positive_total > 0.5:
        extremity_score *= (1.0 - positive_total * 0.6)
    extremity_score = round(min(10.0, max(0.0, extremity_score)), 2)

    # --- 综合计算 ---
    w = STABILITY_FACTOR_WEIGHTS
    final_score = (
        negative_score * w["negative_weight"] +
        entropy_score * w["entropy"] +
        extremity_score * w["extremity"]
    )
    final_score = round(min(10.0, max(0.0, final_score)), 2)

    stability_level, level_color = self._get_stability_level(final_score)

    return {
        "negative_weight_score": negative_score,
        "entropy_score": entropy_score,
        "extremity_score": extremity_score,
        "stability_score": final_score,
        "stability_level": stability_level,
        "stability_color": level_color,
        "factor_weights_source": "专家设定（负面40% + 熵30% + 极端30%），基于情绪维度理论"
    }
```

- [ ] **Step 2: Update `predict` method to include VAD, raw output, sub-scores**

In the `predict` method, after computing `stability_score` (now a dict), update the return block:

```python
# In predict(), replace the old stability_score usage:

# 计算情绪稳定度（现在返回字典）
stability_result = self._calculate_stability_score(probs_dict)
stability_score = stability_result["stability_score"]
stability_level = stability_result["stability_level"]
level_color = stability_result["stability_color"]

# 计算 VAD 维度指标
from vad_dimensions import compute_vad_dimensions
vad_dimensions = compute_vad_dimensions(probs_dict)

# ... (rest of existing logic for mixed_emotions, compound_emotion, etc.)

return {
    "success": True,
    "主要情绪": main_emotion,
    "置信度": round(confidence, 4),
    # 完整8类概率（含"其他"）
    "完整概率_8类": {k: round(v, 4) for k, v in probs_dict.items()},
    # 显示用7类概率（排除"其他"）
    "所有情绪概率": display_probs,
    # 原始模型输出
    "原始模型输出": {
        "labels": raw_labels,
        "scores": [round(float(s), 6) for s in raw_scores] if raw_scores else []
    },
    # 稳定度及分项
    "情绪稳定度分数": stability_score,
    "情绪状态等级": stability_level,
    "等级颜色": level_color,
    "稳定度分项": {
        "negative_weight_score": stability_result["negative_weight_score"],
        "entropy_score": stability_result["entropy_score"],
        "extremity_score": stability_result["extremity_score"],
        "factor_weights_source": stability_result["factor_weights_source"]
    },
    # VAD 维度指标
    "valence_score": vad_dimensions["valence_score"],
    "arousal_score": vad_dimensions["arousal_score"],
    "dominance_score": vad_dimensions["dominance_score"],
    "negative_load": vad_dimensions["negative_load"],
    "emotional_uncertainty": vad_dimensions["emotional_uncertainty"],
    "estimation_note": vad_dimensions["estimation_note"],
    # 原有字段
    "调节建议": advice,
    "混合情绪": mixed_emotions,
    "复合情绪": compound_emotion.get("name", "") if compound_emotion else "",
    "复合情绪详情": compound_emotion if compound_emotion else None,
    "情绪分析摘要": emotion_summary
}
```

- [ ] **Step 3: Update existing tests**

```bash
conda run -n audio python -m pytest test_emotion_algorithm.py test_model_integration.py -v
```

Check that existing tests pass with the modified return structure. Update tests if any assertion checks for old field names.

---

### Task 1.4: Modify `history_manager.py` — Extended schema + research mode

**Files:**
- Modify: `history_manager.py` (extend `_sanitize_record`, `add_record`, add research mode flag)
- Test: `test_history_extended.py` (new)

- [ ] **Step 1: Extend `_sanitize_record` for new fields**

Add new field handling in `_sanitize_record`:

```python
# New fields in _sanitize_record (add after existing fields):
sanitized['valence_score'] = _safe_float(record.get('valence_score'), -1.0, 1.0)  # type: ignore
sanitized['arousal_score'] = _safe_float(record.get('arousal_score'), 0.0, 1.0)  # type: ignore
sanitized['dominance_score'] = _safe_float(record.get('dominance_score'), 0.0, 1.0)  # type: ignore
sanitized['negative_load'] = _safe_float(record.get('negative_load'), 0.0, 1.0)  # type: ignore
sanitized['emotional_uncertainty'] = _safe_float(record.get('emotional_uncertainty'), 0.0, 1.0)  # type: ignore
sanitized['raw_model_output'] = record.get('raw_model_output', {})  # type: ignore
sanitized['probs_8'] = record.get('probs_8', {})  # type: ignore
sanitized['probs_7'] = record.get('probs_7', {})  # type: ignore
sanitized['stability_factors'] = record.get('stability_factors', {})  # type: ignore
sanitized['audio_quality'] = record.get('audio_quality', {})  # type: ignore
sanitized['model_name'] = str(record.get('model_name', ''))[:50]
sanitized['algorithm_version'] = str(record.get('algorithm_version', ''))[:20]
sanitized['app_version'] = str(record.get('app_version', ''))[:20]
sanitized['is_research_mode'] = bool(record.get('is_research_mode', False))
sanitized['assessment_reliability'] = str(record.get('assessment_reliability', ''))[:20]
# estimation_note for transparency
sanitized['estimation_note'] = str(record.get('estimation_note', ''))[:500]
```

Add helper function at module level:

```python
def _safe_float(value, min_val, max_val):
    """安全转换浮点数到指定范围"""
    try:
        v = float(value)
        return max(min_val, min(max_val, v))
    except (TypeError, ValueError):
        return min_val
```

- [ ] **Step 2: Extend `add_record` to save all new fields**

Update the record construction in `add_record`:

```python
record = {
    # ... existing fields ...
    "audio_file": str(result.get("audio_file", "")),
    "main_emotion": main_emotion,
    "anxiety_score": score_f,
    "emotion_level": emotion_level,
    "confidence": conf_f,
    "suggestion": str(result.get("suggestion_text", result.get("调节建议", "")))[:500],
    "compound_emotion": str(result.get("复合情绪", ""))[:20],
    "emotion_summary": str(result.get("情绪分析摘要", ""))[:500],
    # New P0 fields
    "valence_score": result.get("valence_score", 0.0),
    "arousal_score": result.get("arousal_score", 0.0),
    "dominance_score": result.get("dominance_score", 0.0),
    "negative_load": result.get("negative_load", 0.0),
    "emotional_uncertainty": result.get("emotional_uncertainty", 0.0),
    "raw_model_output": result.get("原始模型输出", {}),
    "probs_8": result.get("完整概率_8类", {}),
    "probs_7": result.get("所有情绪概率", {}),
    "stability_factors": result.get("稳定度分项", {}),
    "assessment_reliability": result.get("assessment_reliability", ""),
    "model_name": result.get("model_name", ""),
    "algorithm_version": result.get("algorithm_version", "1.0.0"),
    "app_version": result.get("app_version", "2.0.0"),
    "is_research_mode": bool(result.get("is_research_mode", False)),
    "estimation_note": str(result.get("estimation_note", ""))[:500],
}
```

- [ ] **Step 3: Add research mode disable-cleanup logic**

Modify the auto-cleanup section in `add_record`:

```python
# 科研模式下禁用自动清理
is_research = bool(result.get("is_research_mode", False))
if not is_research:
    # 达到自动清理阈值时，通知并删除最早的记录
    if len(self.records) >= self.AUTO_CLEAN_THRESHOLD:
        # ... existing cleanup logic ...
    # 超过最大记录数时，清理最旧的记录
    if len(self.records) > self.MAX_RECORDS:
        # ... existing cleanup logic ...
```

- [ ] **Step 4: Commit**

```bash
git add emotion_recognizer.py history_manager.py
git commit -m "feat(P0): add stability sub-scores, VAD dimensions, raw output, extended history schema, research mode cleanup disable"
```

---

### Task 1.5: Update `main.py` — Version constants

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add version constants**

After the module docstring in `main.py`:

```python
# 版本常量
APP_VERSION = "2.0.0"
ALGORITHM_VERSION = "2.0.0-p0"
```

- [ ] **Step 2: Pass versions to recognizer results**

In `main.py`, these can be imported by other modules as needed.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat(P0): add APP_VERSION and ALGORITHM_VERSION constants"
```

---

### Task 1.6: Update `gui.py` — Recording quality feedback + result cards (Part 1)

**Files:**
- Modify: `gui.py` (add recording quality display, update result display section)
- Create: `gui_widgets/__init__.py`
- Create: `gui_widgets/result_cards.py`

This is the largest P0 GUI task. Due to `gui.py` being ~2000 lines, I'll create the `result_cards.py` widget module first, then integrate it.

- [ ] **Step 1: Create `gui_widgets/__init__.py`**

```python
# gui_widgets package
from .result_cards import (
    EmotionCard,
    DimensionCard,
    StabilityCard,
    QualityCard,
    ReliabilityCard,
    StructuredReportWidget
)
```

- [ ] **Step 2: Create `gui_widgets/result_cards.py` with card widgets**

Create the structured report card widgets:
- `EmotionCard`: shows main emotion, confidence, 8-class probabilities
- `DimensionCard`: shows VAD dimensions with visual indicators
- `StabilityCard`: shows stability score + sub-factor breakdown
- `QualityCard`: shows audio quality metrics
- `ReliabilityCard`: shows reliability label (high/medium/low)
- `StructuredReportWidget`: composite widget containing all cards

(Full code ~500 lines — to be written during implementation)

- [ ] **Step 3: Add recording quality feedback to recorder panel**

In `gui.py`, add real-time quality indicators during recording:
- Volume level bar with "too low" / "good" / "too high" zones
- Recording duration display
- Quality warnings (auto-detected)

- [ ] **Step 4: Integrate result cards into result display area**

Replace or augment existing result display with `StructuredReportWidget`.

- [ ] **Step 5: Commit**

```bash
git add gui_widgets/ gui.py
git commit -m "feat(P0): add structured assessment report cards, recording quality feedback"
```

---

## Phase 1 P0 Completion Checkpoint

After Tasks 1.1-1.6 are complete, run full test suite and visual verification:

```bash
conda run -n audio python -m pytest test_audio_quality.py test_vad_dimensions.py test_emotion_algorithm.py test_history_extended.py test_model_integration.py -v
```

```bash
# Visual verification - launch and screenshot
conda run -n audio python main.py &
```

Commit and push:
```bash
git add -A
git commit -m "feat(P0): complete core improvements for research-grade assessment"
git push origin feature/research-voice-recognition
```

---

## Phase 2: P1 Research Enhancements (Tasks 2.1-2.6)

### Task 2.1: Create `audio_features.py` — Acoustic feature extraction

**Files:**
- Create: `audio_features.py`
- Test: `test_audio_features.py`

Core features to extract:
- F0 (mean, std, range) using librosa.pyin
- Jitter, Shimmer, HNR using praat-parselmouth (if available) or librosa approximations
- MFCC (mean, std per coefficient)
- Spectral centroid, spectral rolloff
- Speech rate (syllables/sec estimate)
- Silence ratio
- Short-term energy

Psychological state mapping functions:
- `compute_stress_index()`
- `compute_anxiety_index()`
- `compute_depression_tendency_index()`
- `compute_emotional_activation()`
- `compute_speech_stability()`
- `compute_assessment_reliability()`

Requirements addition: `praat-parselmouth>=0.4.0` to `requirements.txt`

---

### Task 2.2: Create `reliability.py` — Reliability assessment

**Files:**
- Create: `reliability.py`
- Test: `test_reliability.py`

Features:
- `compute_icc()`: Intraclass Correlation Coefficient for test-retest
- `evaluate_model_agreement()`: Compare Base vs Large model outputs
- `compute_multi_sample_reliability()`: Mean, std, consistency across 3 samples
- `get_reliability_level()`: High/Medium/Low label

---

### Task 2.3: Create `export_manager.py` — Data export

**Files:**
- Create: `export_manager.py`
- Test: `test_export_manager.py`

Features:
- `export_records_csv()`: Export history to CSV (SPSS/Excel compatible)
- `export_research_dataset()`: Export complete research dataset as JSON
- `export_session_report()`: Export single session as structured JSON

---

### Task 2.4: Update `emotion_recognizer.py` — Compound scoring + dual-model

**Files:**
- Modify: `emotion_recognizer.py`

Changes:
- Replace `_analyze_compound_emotions` with intensity scoring formula
- Add `predict_dual_model()` method for research mode (Base + Large)
- Add `_compute_model_agreement()` helper

---

### Task 2.5: Update `history_manager.py` — Statistics APIs

**Files:**
- Modify: `history_manager.py`

Add methods:
- `get_statistics()`: summary statistics
- `get_emotion_distribution()`: emotion frequency distribution
- `get_stability_summary()`: stability mean, std, trend
- `get_compound_emotion_stats()`: compound emotion frequency
- `export_records_csv(output_path)`: CSV export
- `export_research_dataset(output_path)`: JSON research dataset

---

### Task 2.6: GUI upgrades — Research mode + Radar chart + Stats panel

**Files:**
- Create: `gui_widgets/research_panel.py`
- Create: `gui_widgets/stats_panel.py`
- Modify: `gui.py`

Changes:
- Research mode toggle (quick vs research)
- Research mode flow: noise check → prompt recording → quality gate → analysis
- VAD dimension radar chart (matplotlib)
- History statistics panel (emotion distribution, stability trend, compound stats)
- Export buttons (CSV, JSON)
- Multi-sample recording progress (1/3, 2/3, 3/3)

---

## Phase 3: P2 Advanced Features (Tasks 3.1-3.5)

### Task 3.1: Create `baseline.py` — Personal baseline

**Files:**
- Create: `baseline.py`
- Create: `gui_widgets/baseline_panel.py`
- Test: `test_baseline.py`

---

### Task 3.2: Create `statistics.py` — Statistical analysis

**Files:**
- Create: `statistics.py`
- Test: `test_statistics.py`

---

### Task 3.3: Anomaly detection + temporal analysis

**Files:**
- Modify: `statistics.py` (extend)
- Modify: `gui_widgets/stats_panel.py` (extend)

---

### Task 3.4: Data-driven weight optimization framework

**Files:**
- Modify: `emotion_recognizer.py` (add weight optimization notes, framework)

---

### Task 3.5: Standardized report generation

**Files:**
- Modify: `export_manager.py` (add report generation)
- Modify: `gui.py` (add report preview)

---

## Final Completion Checklist

- [ ] All P0 tests pass
- [ ] All P1 tests pass
- [ ] All P2 tests pass
- [ ] No regression in existing tests
- [ ] GUI launches without errors
- [ ] Recording → Analysis → Save flow works end-to-end
- [ ] Research mode flow works end-to-end
- [ ] Visual verification: result cards layout correct
- [ ] Visual verification: radar chart renders
- [ ] Visual verification: stats panel displays correctly
- [ ] CSV/JSON export produces valid files
- [ ] Research mode disables auto-cleanup
- [ ] Version metadata saved with each record
- [ ] VAD estimation note included in output
- [ ] Backward compatibility: old history records load correctly
