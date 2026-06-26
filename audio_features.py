# -*- coding: utf-8 -*-
"""
语音情绪识别系统 - 声学特征提取模块 (P1)

本模块负责从音频信号中提取临床级声学特征，用于心理状态评估。
特征包括时间域、频率域和声带级别的指标。

特征列表：
  时间/能量特征：
    - 平均音量 (rms_mean)
    - 音量方差 (rms_std)
    - 短时能量 (short_term_energy)
    - 语速估计 (speech_rate)
    - 静音比例 (silence_ratio)

  频率特征：
    - 平均音高 F0 (f0_mean)
    - F0 标准差 (f0_std)
    - F0 范围 (f0_range)
    - 频谱质心 (spectral_centroid)

  声带/音质特征（使用 praat-parselmouth 或 librosa 近似）：
    - Jitter (频率微扰)
    - Shimmer (振幅微扰)
    - HNR (谐噪比)

  MFCC 特征：
    - MFCC 均值和标准差（每系数）

心理状态映射指标：
    - stress_index: 压力指数
    - anxiety_index: 焦虑指数
    - depression_tendency_index: 低落倾向
    - emotional_activation: 情绪激活度
    - speech_stability: 语音稳定性

参考文献：
  - Jitter/Shimmer/HNR 是语音临床分析的成熟指标 (Teixeira et al., 2013)
  - F0 与情绪激活度显著相关 (Banse & Scherer, 1996)
  - 语速与焦虑/低落状态相关 (Mundt et al., 2007)

作者：Jiawei Li
许可证：GPL v3
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def extract_acoustic_features(filepath, sr=16000):
    """
    从音频文件提取完整的声学特征集

    参数：
        filepath (str): 音频文件路径
        sr (int): 目标采样率

    返回值：
        dict or None: 声学特征字典
    """
    try:
        from audio_quality import load_audio
        audio, actual_sr = load_audio(filepath, target_sr=sr)
        if audio is None or len(audio) == 0:
            return None

        features = {}

        # ---- 时间/能量特征 ----
        features.update(_extract_temporal_features(audio, actual_sr))

        # ---- 频率特征 (F0, spectral) ----
        features.update(_extract_frequency_features(audio, actual_sr))

        # ---- 声带特征 (jitter, shimmer, HNR) ----
        features.update(_extract_voice_quality_features(filepath, audio, actual_sr))

        # ---- MFCC 特征 ----
        features.update(_extract_mfcc_features(audio, actual_sr))

        return features

    except Exception as e:
        logger.error(f"声学特征提取失败: {e}")
        return None


def _extract_temporal_features(audio, sr):
    """提取时间和能量域特征"""
    features = {}

    # RMS 能量
    rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    frame_len = 2048
    hop = 512
    n_frames = max(1, (len(audio) - frame_len) // hop + 1)
    rms_frames = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        end = min(start + frame_len, len(audio))
        chunk = audio[start:end]
        rms_frames[i] = np.sqrt(np.mean(chunk ** 2) + 1e-10)

    features['rms_mean'] = round(float(rms), 6)
    features['rms_std'] = round(float(np.std(rms_frames)), 6)

    # 短时能量（归一化）
    short_term_energy = float(np.mean(rms_frames))
    features['short_term_energy'] = round(short_term_energy, 6)

    # 静音比例
    from audio_quality import detect_speech_segments
    segments = detect_speech_segments(audio, sr, energy_threshold=max(0.005, rms * 0.2))
    speech_duration = sum(end - start for start, end in segments)
    total_duration = len(audio) / sr
    features['silence_ratio'] = round(1.0 - speech_duration / total_duration if total_duration > 0 else 1.0, 4)

    # 语速估计（每秒音节数近似：基于能量包络的峰值计数）
    try:
        # 简单方法：使用能量包络的过零率
        import librosa
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr,
                                            backtrack=True)
        n_syllables = len(onsets)
        speech_rate = n_syllables / speech_duration if speech_duration > 0.5 else 0.0
        features['speech_rate'] = round(speech_rate, 2)
        features['n_syllables'] = n_syllables
    except Exception:
        features['speech_rate'] = 0.0
        features['n_syllables'] = 0

    return features


def _extract_frequency_features(audio, sr):
    """提取频率域特征（F0, 频谱质心）"""
    features = {}

    try:
        import librosa

        # F0 提取 (使用 PYIN 算法，更精确)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),   # ~65 Hz
            fmax=librosa.note_to_hz('C7'),    # ~2093 Hz
            sr=sr
        )

        # 过滤未浊化部分
        if f0 is not None:
            voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
            if len(voiced_f0) > 0:
                features['f0_mean'] = round(float(np.mean(voiced_f0)), 2)
                features['f0_std'] = round(float(np.std(voiced_f0)), 2)
                features['f0_min'] = round(float(np.min(voiced_f0)), 2)
                features['f0_max'] = round(float(np.max(voiced_f0)), 2)
                features['f0_range'] = round(features['f0_max'] - features['f0_min'], 2)
                features['voiced_ratio'] = round(len(voiced_f0) / len(f0) if len(f0) > 0 else 0.0, 4)
            else:
                features.update(_default_f0_values())

        # 频谱质心
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = round(float(np.mean(centroid)), 2)
        features['spectral_centroid_std'] = round(float(np.std(centroid)), 2)

        # 频谱滚降
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spectral_rolloff_mean'] = round(float(np.mean(rolloff)), 2)

    except Exception as e:
        logger.warning(f"频率特征提取失败，使用默认值: {e}")
        features.update(_default_f0_values())
        features['spectral_centroid_mean'] = 0.0
        features['spectral_centroid_std'] = 0.0
        features['spectral_rolloff_mean'] = 0.0

    return features


def _default_f0_values():
    return {
        'f0_mean': 0.0, 'f0_std': 0.0, 'f0_min': 0.0,
        'f0_max': 0.0, 'f0_range': 0.0, 'voiced_ratio': 0.0
    }


def _extract_voice_quality_features(filepath, audio, sr):
    """提取声带音质特征（jitter, shimmer, HNR）"""
    features = {
        'jitter_local': 0.0,
        'jitter_ppq5': 0.0,
        'shimmer_local': 0.0,
        'shimmer_apq11': 0.0,
        'hnr_mean': 0.0
    }

    # 优先使用 praat-parselmouth（专业语音分析工具）
    try:
        import parselmouth
        try:
            snd = parselmouth.Sound(filepath)
        except Exception:
            # 回退：根据音频数据创建 Sound 对象
            snd = parselmouth.Sound(audio.tolist(), sampling_frequency=sr)

        # F0 提取（Praat 算法）
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
        f0_values = pitch.selected_array['frequency']
        voiced_f0 = f0_values[f0_values > 0]

        if len(voiced_f0) > 0:
            # 如果 librosa 没有提取到 F0，则用 Praat 结果
            if features.get('f0_mean', 0) == 0:
                features['f0_mean'] = round(float(np.mean(voiced_f0)), 2)
                features['f0_std'] = round(float(np.std(voiced_f0)), 2)

        # PointProcess (声门脉冲)
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 600)

        # Jitter (local)
        try:
            jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            features['jitter_local'] = round(float(jitter), 6)
        except Exception:
            pass

        # Jitter (ppq5)
        try:
            jitter_ppq5 = parselmouth.praat.call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            features['jitter_ppq5'] = round(float(jitter_ppq5), 6)
        except Exception:
            pass

        # Shimmer (local)
        try:
            shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features['shimmer_local'] = round(float(shimmer), 6)
        except Exception:
            pass

        # Shimmer (apq11)
        try:
            shimmer_apq11 = parselmouth.praat.call([snd, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features['shimmer_apq11'] = round(float(shimmer_apq11), 6)
        except Exception:
            pass

        # HNR (谐噪比)
        try:
            hnr = snd.to_harmonicity_cc()
            if hnr is not None:
                hnr_values = hnr.values[hnr.values != -200]
                if len(hnr_values) > 0:
                    features['hnr_mean'] = round(float(np.mean(hnr_values)), 2)
        except Exception:
            pass

    except ImportError:
        logger.debug("praat-parselmouth 未安装，使用 librosa 近似方法")
        features.update(_approx_voice_quality_librosa(audio, sr))

    return features


def _approx_voice_quality_librosa(audio, sr):
    """使用 librosa 近似计算声带特征（回退方案）"""
    features = {}
    try:
        # 使用谐波-冲击源分离估计 HNR
        import librosa
        y_harmonic, y_percussive = librosa.effects.hpss(audio)
        harmonic_power = np.sum(y_harmonic ** 2)
        percussive_power = np.sum(y_percussive ** 2) + 1e-10
        features['hnr_mean'] = round(float(10 * np.log10(harmonic_power / percussive_power)), 2)

        # Jitter 近似（F0 周期变异性）
        # 简单方法：基于过零率波动
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=512)[0]
        features['jitter_local'] = round(float(np.std(zcr) / (np.mean(zcr) + 1e-10)), 6)

        # Shimmer 近似（振幅变异性）
        rms_frames = []
        for i in range(0, len(audio) - 2048, 512):
            chunk = audio[i:i+2048]
            rms_frames.append(np.sqrt(np.mean(chunk ** 2) + 1e-10))
        if len(rms_frames) > 1:
            rms_arr = np.array(rms_frames)
            features['shimmer_local'] = round(float(np.std(rms_arr) / (np.mean(rms_arr) + 1e-10)), 6)
        else:
            features['shimmer_local'] = 0.0
    except Exception:
        features['hnr_mean'] = 0.0
        features['jitter_local'] = 0.0
        features['shimmer_local'] = 0.0

    return features


def _extract_mfcc_features(audio, sr):
    """提取 MFCC 特征"""
    features = {}
    try:
        import librosa
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = round(float(np.mean(mfcc[i])), 4)
            features[f'mfcc_{i+1}_std'] = round(float(np.std(mfcc[i])), 4)
    except Exception:
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = 0.0
            features[f'mfcc_{i+1}_std'] = 0.0
    return features


# ===========================================================================
# 声音-心理状态映射函数
# 基于文献和专家知识的加权映射，非临床诊断
# ===========================================================================

def compute_stress_index(acoustic_features, vad_dimensions=None):
    """
    压力指数估计

    依据：高唤醒 + 负性情绪 + 音量/音高波动

    参数：
        acoustic_features (dict): 声学特征
        vad_dimensions (dict): VAD 维度（可选）

    返回值：
        float: 压力指数 (0.0-1.0)
    """
    score = 0.0

    # F0 波动贡献
    f0_std = acoustic_features.get('f0_std', 0)
    f0_std_norm = min(1.0, f0_std / 80.0)  # 归一化到 ~80Hz
    score += f0_std_norm * 0.25

    # 音量方差贡献
    rms_std = acoustic_features.get('rms_std', 0)
    rms_std_norm = min(1.0, rms_std / 0.1)
    score += rms_std_norm * 0.20

    # 语速贡献（高语速→高压力）
    speech_rate = acoustic_features.get('speech_rate', 3)
    rate_norm = min(1.0, max(0.0, (speech_rate - 2) / 6))
    score += rate_norm * 0.15

    # VAD 维度贡献
    if vad_dimensions:
        arousal = vad_dimensions.get('arousal_score', 0.5)
        negative_load = vad_dimensions.get('negative_load', 0.0)
        score += arousal * 0.20
        score += negative_load * 0.20

    return round(min(1.0, max(0.0, score)), 4)


def compute_anxiety_index(acoustic_features, vad_dimensions=None):
    """
    焦虑指数估计

    依据：恐惧 + 悲伤 + 高唤醒 + 停顿/语速异常

    参数：
        acoustic_features (dict): 声学特征
        vad_dimensions (dict): VAD 维度（可选）

    返回值：
        float: 焦虑指数 (0.0-1.0)
    """
    score = 0.0

    # F0 升高（紧张）
    f0_mean = acoustic_features.get('f0_mean', 120)
    f0_norm = min(1.0, max(0.0, (f0_mean - 100) / 150))
    score += f0_norm * 0.15

    # 静音比例高（犹豫、停顿多）
    silence = acoustic_features.get('silence_ratio', 0.3)
    score += silence * 0.20

    # jitter 高（声带紧张）
    jitter = acoustic_features.get('jitter_local', 0)
    jitter_norm = min(1.0, jitter * 50)
    score += jitter_norm * 0.20

    # VAD 维度
    if vad_dimensions:
        arousal = vad_dimensions.get('arousal_score', 0.5)
        score += arousal * 0.25
        dominance = vad_dimensions.get('dominance_score', 0.5)
        score += (1.0 - dominance) * 0.20

    return round(min(1.0, max(0.0, score)), 4)


def compute_depression_tendency_index(acoustic_features, vad_dimensions=None):
    """
    低落倾向指数估计

    依据：悲伤 + 低唤醒 + 低音量 + 长静音

    参数：
        acoustic_features (dict): 声学特征
        vad_dimensions (dict): VAD 维度（可选）

    返回值：
        float: 低落倾向 (0.0-1.0)
    """
    score = 0.0

    # 低音量
    rms_mean = acoustic_features.get('rms_mean', 0.1)
    rms_norm = max(0.0, 1.0 - rms_mean / 0.15)
    score += rms_norm * 0.15

    # 低语速
    speech_rate = acoustic_features.get('speech_rate', 3)
    rate_norm = max(0.0, 1.0 - speech_rate / 5)
    score += rate_norm * 0.15

    # 高静音比例
    silence = acoustic_features.get('silence_ratio', 0.3)
    score += silence * 0.20

    # 低 F0 变异性（单调）
    f0_std = acoustic_features.get('f0_std', 30)
    f0_std_norm = max(0.0, 1.0 - f0_std / 50)
    score += f0_std_norm * 0.15

    # VAD 维度
    if vad_dimensions:
        arousal = vad_dimensions.get('arousal_score', 0.5)
        score += (1.0 - arousal) * 0.20
        negative_load = vad_dimensions.get('negative_load', 0.0)
        score += negative_load * 0.15

    return round(min(1.0, max(0.0, score)), 4)


def compute_emotional_activation(acoustic_features, vad_dimensions=None):
    """
    情绪激活度估计

    依据：唤醒度 + 短时能量 + F0 波动

    参数：
        acoustic_features (dict): 声学特征
        vad_dimensions (dict): VAD 维度（可选）

    返回值：
        float: 情绪激活度 (0.0-1.0)
    """
    score = 0.0

    # 短时能量
    energy = acoustic_features.get('short_term_energy', 0.05)
    energy_norm = min(1.0, energy / 0.3)
    score += energy_norm * 0.25

    # F0 均值（高音高 = 高激活）
    f0_mean = acoustic_features.get('f0_mean', 120)
    f0_norm = min(1.0, f0_mean / 300)
    score += f0_norm * 0.20

    # F0 波动
    f0_std = acoustic_features.get('f0_std', 30)
    f0_std_norm = min(1.0, f0_std / 60)
    score += f0_std_norm * 0.20

    # VAD
    if vad_dimensions:
        arousal = vad_dimensions.get('arousal_score', 0.5)
        score += arousal * 0.35

    return round(min(1.0, max(0.0, score)), 4)


def compute_speech_stability(acoustic_features):
    """
    语音稳定性估计

    依据：jitter、shimmer、HNR、音量波动

    参数：
        acoustic_features (dict): 声学特征

    返回值：
        float: 语音稳定性 (0.0-1.0，越高越稳定)
    """
    score = 1.0

    # Jitter（越小越稳定）
    jitter = acoustic_features.get('jitter_local', 0)
    jitter_penalty = min(0.3, jitter * 5)
    score -= jitter_penalty

    # Shimmer（越小越稳定）
    shimmer = acoustic_features.get('shimmer_local', 0)
    shimmer_penalty = min(0.3, shimmer * 5)
    score -= shimmer_penalty

    # HNR（越高越稳定）
    hnr = acoustic_features.get('hnr_mean', 20)
    if hnr < 15:
        score -= 0.2
    elif hnr > 25:
        score += 0.1

    return round(min(1.0, max(0.0, score)), 4)


def compute_psychological_indicators(acoustic_features, vad_dimensions=None):
    """
    从声学特征和 VAD 维度计算所有心理状态指标

    参数：
        acoustic_features (dict): 声学特征字典
        vad_dimensions (dict): VAD 维度字典（可选）

    返回值：
        dict: 心理状态指标字典
    """
    if acoustic_features is None:
        acoustic_features = {}

    return {
        "stress_index": compute_stress_index(acoustic_features, vad_dimensions),
        "anxiety_index": compute_anxiety_index(acoustic_features, vad_dimensions),
        "depression_tendency_index": compute_depression_tendency_index(acoustic_features, vad_dimensions),
        "emotional_activation": compute_emotional_activation(acoustic_features, vad_dimensions),
        "speech_stability": compute_speech_stability(acoustic_features),
    }
