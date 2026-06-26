# -*- coding: utf-8 -*-
"""
语音情绪识别系统 - 可靠性评估模块 (P1)

本模块提供科研级的可靠性评估功能：
1. 测试-重测信度 (ICC - Intraclass Correlation Coefficient)
2. 双模型一致性检验（模型规模敏感性检验）
3. 多次采样综合评估（均值、标准差、一致性）
4. 综合可靠性等级评定

⚠️ 注意：
  - 双模型比较（Base vs Large）共享训练数据与架构，一致性仅反映模型规模
    不敏感性，不代表真正可靠性。已重新定义为"模型规模敏感性检验"。
  - ICC 计算需要同一被试多次测量数据。

作者：Jiawei Li
许可证：GPL v3
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# 双模型一致性检验的定义说明
MODEL_AGREEMENT_NOTE = (
    "⚠️ 模型规模敏感性检验：Base与Large共享训练数据与架构，"
    "一致性仅反映模型规模不敏感，不等于结果绝对可靠。"
    "P2阶段建议引入异构模型（HuBERT/wav2vec2-based）作为真正的对照验证。"
)


def compute_icc(values_across_sessions):
    """
    计算组内相关系数 (ICC, two-way random, single measures)

    用于评估测试-重测信度。需要同一被试多次测量的相同指标。

    参数：
        values_across_sessions (list of list): 每个session的测量值列表
            例如 [[2.1, 2.3, 1.9], [2.2, 2.4, 2.0], ...]

    返回值：
        float: ICC 值 (-1.0 ~ 1.0)，越高表示信度越好
    """
    if not values_across_sessions or len(values_across_sessions) < 2:
        return 0.0

    try:
        # 转换为 numpy 数组
        data = np.array(values_across_sessions, dtype=float)
        n_sessions, n_measures = data.shape

        if n_sessions < 2 or n_measures < 2:
            return 0.0

        # 总均值
        grand_mean = np.mean(data)

        # 组间均方 (MSB)
        session_means = np.mean(data, axis=1)
        msb = n_measures * np.sum((session_means - grand_mean) ** 2) / (n_sessions - 1) if n_sessions > 1 else 0.0

        # 组内均方 (MSW)
        msw = 0.0
        for i in range(n_sessions):
            msw += np.sum((data[i] - session_means[i]) ** 2)
        msw /= (n_sessions * (n_measures - 1)) if n_measures > 1 else 1.0

        if msb + msw < 1e-10:
            return 0.0

        icc = (msb - msw) / (msb + msw)
        return round(max(-1.0, min(1.0, icc)), 4)

    except Exception as e:
        logger.error(f"ICC计算失败: {e}")
        return 0.0


def evaluate_model_agreement(result_base, result_large):
    """
    评估 Base 和 Large 模型的一致性（模型规模敏感性检验）

    比较两个模型在主要情绪、稳定度、VAD维度上的差异。

    参数：
        result_base (dict): Base 模型的预测结果
        result_large (dict): Large 模型的预测结果

    返回值：
        dict: 一致性评估结果
    """
    if not result_base or not result_large:
        return {"agreement": 0.0, "level": "低", "note": MODEL_AGREEMENT_NOTE}

    try:
        agreements = []

        # 主要情绪一致
        if result_base.get('主要情绪') == result_large.get('主要情绪'):
            agreements.append(1.0)
        else:
            agreements.append(0.0)

        # 稳定度分数差异（< 1.0 分认为一致）
        score_diff = abs(
            result_base.get('情绪稳定度分数', 5.0) -
            result_large.get('情绪稳定度分数', 5.0)
        )
        agreements.append(max(0.0, 1.0 - score_diff / 3.0))

        # VAD 维度差异
        for key in ['valence_score', 'arousal_score', 'dominance_score']:
            diff = abs(
                result_base.get(key, 0.0) - result_large.get(key, 0.0)
            )
            agreements.append(max(0.0, 1.0 - diff * 2.0))

        overall_agreement = round(float(np.mean(agreements)), 4)
        level = _agreement_level(overall_agreement)

        return {
            "agreement": overall_agreement,
            "level": level,
            "emotion_match": result_base.get('主要情绪') == result_large.get('主要情绪'),
            "primary_model": result_base.get('model_name', ''),
            "secondary_model": result_large.get('model_name', ''),
            "note": MODEL_AGREEMENT_NOTE
        }
    except Exception as e:
        logger.error(f"模型一致性评估失败: {e}")
        return {"agreement": 0.0, "level": "低", "note": MODEL_AGREEMENT_NOTE}


def _agreement_level(score):
    """将一致性分数转为等级"""
    if score >= 0.8:
        return "高"
    elif score >= 0.5:
        return "中"
    else:
        return "低"


def evaluate_multi_sample_reliability(results_list):
    """
    多次采样综合评估

    参数：
        results_list (list): 多次采样的预测结果列表（2-5次）

    返回值：
        dict: 综合评估结果
    """
    if not results_list or len(results_list) < 1:
        return {"consistency": 0.0, "reliability": "低", "note": "样本不足"}

    try:
        n = len(results_list)

        # 提取关键指标
        stability_scores = [r.get('情绪稳定度分数', 0.0) for r in results_list]
        main_emotions = [r.get('主要情绪', '') for r in results_list]
        valence_scores = [r.get('valence_score', 0.0) for r in results_list]
        arousal_scores = [r.get('arousal_score', 0.0) for r in results_list]

        # 主要情绪一致性
        from collections import Counter
        emotion_counts = Counter(main_emotions)
        most_common_emotion, most_common_count = emotion_counts.most_common(1)[0]
        emotion_consistency = most_common_count / n

        # 稳定度统计
        stability_mean = round(float(np.mean(stability_scores)), 2)
        stability_std = round(float(np.std(stability_scores)), 2)

        # VAD 维度统计
        valence_mean = round(float(np.mean(valence_scores)), 4)
        valence_std = round(float(np.std(valence_scores)), 4)
        arousal_mean = round(float(np.mean(arousal_scores)), 4)
        arousal_std = round(float(np.std(arousal_scores)), 4)

        # 综合一致性 (CV-based)
        cv_stability = stability_std / max(stability_mean, 0.01)
        cv_valence = valence_std / max(abs(valence_mean) + 0.1, 0.01)
        consistency = 1.0 - min(1.0, (cv_stability + cv_valence) / 4.0)
        consistency = round(max(0.0, consistency), 4)

        reliability_level = "高" if consistency >= 0.7 else "中" if consistency >= 0.4 else "低"

        return {
            "n_samples": n,
            "session_emotion": most_common_emotion,
            "emotion_consistency": round(emotion_consistency, 2),
            "stability_mean": stability_mean,
            "stability_std": stability_std,
            "valence_mean": valence_mean,
            "valence_std": valence_std,
            "arousal_mean": arousal_mean,
            "arousal_std": arousal_std,
            "consistency": consistency,
            "reliability": reliability_level,
            "note": f"基于 {n} 次采样的综合评估"
        }

    except Exception as e:
        logger.error(f"多次采样评估失败: {e}")
        return {"consistency": 0.0, "reliability": "低", "note": str(e)}


def get_reliability_level(assessment):
    """
    获取综合可靠性等级

    综合考虑：音频质量、模型置信度、多次采样一致性（如有）

    参数：
        assessment (dict): 包含 audio_quality、confidence、consistency 的评估字典

    返回值：
        str: "高" / "中" / "低"
    """
    scores = []

    # 音频质量分数
    audio_q = assessment.get('audio_quality', {})
    if isinstance(audio_q, dict):
        q_score = audio_q.get('quality_score', 0.5)
        scores.append(q_score)

    # 模型置信度
    confidence = assessment.get('confidence', 0.5)
    scores.append(confidence)

    # 多次采样一致性
    consistency = assessment.get('consistency', None)
    if consistency is not None:
        scores.append(consistency)

    if not scores:
        return "低"

    avg = np.mean(scores)
    if avg >= 0.7:
        return "高"
    elif avg >= 0.4:
        return "中"
    return "低"
