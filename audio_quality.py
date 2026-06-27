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
    except Exception:
        pass

    # 回退：使用 wave + scipy
    try:
        import scipy.io.wavfile as wavfile
        from scipy import signal as scipy_signal
        sr, data = wavfile.read(filepath)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        else:
            data = data.astype(np.float32)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        if sr != target_sr:
            data = scipy_signal.resample(data, int(len(data) * target_sr / sr))
        return data.astype(np.float32), target_sr
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
        # If trimmed result is still essentially silence, return empty
        if len(trimmed) > 0:
            rms_trimmed = np.sqrt(np.mean(trimmed ** 2))
            if rms_trimmed < 1e-6:
                return np.array([], dtype=np.float32)
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

        non_silent_indices = np.where(non_silent)[0]
        first = non_silent_indices[0]
        last = non_silent_indices[-1]

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
    speech_start = 0.0

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
            speech_mask = np.zeros(len(audio), dtype=bool)
            for start, end in segments:
                speech_mask[int(start * sr):int(end * sr)] = True
            non_speech = audio[~speech_mask]
            if len(non_speech) > 0:
                noise_rms = float(np.sqrt(np.mean(non_speech ** 2) + 1e-10))
            else:
                noise_rms = 0.0
        else:
            noise_rms = 0.0 if speech_ratio > 0.9 else float(rms)

        noise_level = noise_rms / max(rms, 1e-8)
        noise_level = min(1.0, noise_level)

        # 综合质量分数
        speech_score = min(1.0, speech_ratio / 0.6)
        clip_score = max(0.0, 1.0 - clipping_ratio / AUDIO_QUALITY_THRESHOLDS["max_clipping_ratio"])
        noise_score = max(0.0, 1.0 - noise_level / AUDIO_QUALITY_THRESHOLDS["max_noise_level"])
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

            original_duration = len(audio) / sr

            # 计算质量指标（基于原始音频）
            quality = compute_audio_quality(filepath)
            if quality is None:
                return None

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
