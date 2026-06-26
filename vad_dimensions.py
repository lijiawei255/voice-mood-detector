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

映射依据（专家设定，基于情绪维度理论）：
- 效价：正面情绪（开心）= 正向，负面情绪（愤怒/恐惧/悲伤/厌恶）= 负向
- 唤醒度：高唤醒（愤怒/恐惧/开心/惊讶）> 低唤醒（悲伤/平静/厌恶）
- 掌控感：高掌控（开心/平静/愤怒）> 低掌控（恐惧/悲伤/惊讶）

作者：Jiawei Li
许可证：GPL v3
"""

import numpy as np
import math

# 标注说明（所有VAD输出中包含此字段，确保科研透明性）
VAD_ESTIMATION_NOTE = (
    "⚠️ 从离散情绪概率推导的估计值（非直接连续维度测量），"
    "不作为独立科研变量的唯一来源。建议结合标准化心理量表使用。"
)

# ---------------------------------------------------------------------------
# 效价映射权重（基于情绪效价维度理论）
# 正值 = 正面情绪，负值 = 负面情绪
# ---------------------------------------------------------------------------
VALENCE_WEIGHTS = {
    "愤怒": -0.8,
    "厌恶": -0.7,
    "恐惧": -0.9,
    "开心": 0.9,
    "平静": 0.3,
    "悲伤": -0.8,
    "惊讶": 0.0,   # 惊讶可为正可为负，取中性
    "其他": 0.0
}

# ---------------------------------------------------------------------------
# 唤醒度映射权重（基于情绪激活水平理论）
# 高值 = 高激活能量，低值 = 低激活能量
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 掌控感映射权重（基于情绪控制维度理论）
# 高值 = 高主观掌控感
# ---------------------------------------------------------------------------
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

    综合负面情绪（愤怒、恐惧、悲伤、厌恶）的加权强度。

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
