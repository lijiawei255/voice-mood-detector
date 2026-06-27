# -*- coding: utf-8 -*-
"""
语音情绪识别系统 - 统计分析模块 (P2)

本模块提供科研级的统计分析功能：
1. 效应量计算 (Cohen's d, Hedges' g)
2. 描述性统计
3. 时序心理状态分析（趋势检测）
4. 异常波动检测
5. 情绪分布比较

用于评估干预效果、个体差异和纵向变化。

参考文献：
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
- Hedges, L. V. (1981). Distribution theory for Glass's estimator of effect size.
- Jacobson & Truax (1991). Clinical significance.

作者：Jiawei Li
许可证：GPL v3
"""

import numpy as np
import logging
from collections import Counter

logger = logging.getLogger(__name__)


# ===========================================================================
# 效应量计算
# ===========================================================================

def cohens_d(group1, group2, paired=False):
    """
    计算 Cohen's d 效应量

    参数：
        group1 (list): 第一组数据（如干预前）
        group2 (list): 第二组数据（如干预后）
        paired (bool): 是否为配对样本

    返回值：
        dict: {d, interpretation, magnitude}
    """
    try:
        g1 = np.array(group1, dtype=float)
        g2 = np.array(group2, dtype=float)

        if len(g1) < 2 or len(g2) < 2:
            return {"d": 0.0, "interpretation": "样本量不足", "magnitude": "N/A",
                    "mean_diff": 0.0, "pooled_sd": 0.0}

        mean_diff = np.mean(g2) - np.mean(g1)

        if paired:
            # 配对样本：使用差值的标准差
            diff = g2 - g1
            pooled_sd = np.std(diff, ddof=1)
        else:
            # 独立样本：池化标准差
            n1, n2 = len(g1), len(g2)
            s1 = np.std(g1, ddof=1)
            s2 = np.std(g2, ddof=1)
            pooled_sd = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))

        if pooled_sd < 1e-10:
            d = 0.0
        else:
            d = mean_diff / pooled_sd

        return {
            "d": round(d, 4),
            "interpretation": _interpret_cohens_d(d),
            "magnitude": _magnitude_label(d),
            "mean_diff": round(mean_diff, 4),
            "pooled_sd": round(pooled_sd, 4),
        }
    except Exception as e:
        logger.error(f"Cohen's d 计算失败: {e}")
        return {"d": 0.0, "interpretation": "计算失败", "magnitude": "N/A",
                "mean_diff": 0.0, "pooled_sd": 0.0}


def _interpret_cohens_d(d):
    """解释 Cohen's d 效应量"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "可忽略的效应"
    elif abs_d < 0.5:
        return "小效应"
    elif abs_d < 0.8:
        return "中等效应"
    else:
        return "大效应"


def _magnitude_label(d):
    """效应量标签"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def hedges_g(group1, group2, paired=False):
    """
    计算 Hedges' g（小样本修正的效应量）

    对于小样本（n < 20），Hedges' g 比 Cohen's d 更准确。

    参数：
        group1 (list): 第一组数据
        group2 (list): 第二组数据
        paired (bool): 是否为配对样本

    返回值：
        dict: {g, d, correction_factor, interpretation}
    """
    result = cohens_d(group1, group2, paired=paired)
    d = result["d"]

    n1, n2 = len(group1), len(group2)
    df = n1 + n2 - 2
    if df > 1:
        # Hedges' g 修正因子
        correction = 1.0 - 3.0 / (4.0 * df - 1.0)
        g = d * correction
    else:
        correction = 1.0
        g = d

    return {
        "g": round(g, 4),
        "d": d,
        "correction_factor": round(correction, 4),
        "interpretation": _interpret_cohens_d(g),
        "magnitude": _magnitude_label(g),
    }


# ===========================================================================
# 描述性统计
# ===========================================================================

def descriptive_stats(values):
    """
    计算描述性统计

    参数：
        values (list): 数值列表

    返回值：
        dict: {mean, std, median, min, max, range, skewness, kurtosis, n}
    """
    if not values:
        return {"n": 0}

    try:
        arr = np.array(values, dtype=float)
        n = len(arr)

        from scipy import stats as scipy_stats

        return {
            "n": n,
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr, ddof=1)), 4),
            "median": round(float(np.median(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "range": round(float(np.max(arr) - np.min(arr)), 4),
            "skewness": round(float(scipy_stats.skew(arr)), 4) if n >= 3 else 0,
            "kurtosis": round(float(scipy_stats.kurtosis(arr)), 4) if n >= 4 else 0,
        }
    except Exception as e:
        logger.error(f"描述性统计计算失败: {e}")
        return {"n": 0}


# ===========================================================================
# 时序分析
# ===========================================================================

def detect_trend(timestamps, values, window_size=5):
    """
    检测时序数据的趋势

    使用滑动窗口线性回归检测趋势方向和强度。

    参数：
        timestamps (list): 时间戳列表（字符串 "YYYY-MM-DD HH:MM:SS" 或 datetime）
        values (list): 对应的数值列表
        window_size (int): 滑动窗口大小

    返回值：
        dict: {trend_direction, trend_slope, recent_trend, volatility}
    """
    if len(values) < 3:
        return {"trend_direction": "数据不足", "trend_slope": 0, "recent_trend": "N/A", "volatility": 0}

    try:
        arr = np.array(values, dtype=float)
        n = len(arr)

        # 整体线性趋势
        x = np.arange(n)
        slope, intercept = np.polyfit(x, arr, 1)

        # 最近窗口趋势
        recent_n = min(window_size, n)
        recent_x = np.arange(recent_n)
        recent_y = arr[-recent_n:]
        recent_slope, _ = np.polyfit(recent_x, recent_y, 1)

        # 波动率（标准化标准差）
        volatility = float(np.std(arr) / (np.mean(arr) + 1e-10))

        # 趋势方向
        std_arr = np.std(arr)
        if std_arr < 1e-10 or abs(slope) < 0.01 * std_arr:
            direction = "稳定"
        elif slope > 0:
            direction = "上升"
        else:
            direction = "下降"

        # 最近趋势
        if std_arr < 1e-10 or abs(recent_slope) < 0.01 * std_arr:
            recent_direction = "稳定"
        elif recent_slope > 0:
            recent_direction = "上升"
        else:
            recent_direction = "下降"

        return {
            "trend_direction": direction,
            "trend_slope": round(float(slope), 6),
            "recent_trend": recent_direction,
            "recent_slope": round(float(recent_slope), 6),
            "volatility": round(volatility, 4),
            "n_points": n,
            "trend_strength": round(abs(slope) / (np.std(arr) + 1e-10), 4),
        }
    except Exception as e:
        logger.error(f"趋势检测失败: {e}")
        return {"trend_direction": "计算失败", "trend_slope": 0, "recent_trend": "N/A", "volatility": 0}


# ===========================================================================
# 异常检测
# ===========================================================================

def detect_anomalies(values, method="zscore", threshold=2.5):
    """
    检测异常值

    参数：
        values (list): 数值列表
        method (str): 检测方法 — "zscore" 或 "iqr"
        threshold (float): 异常判定阈值（zscore 默认 2.5，IQR 默认 1.5）

    返回值：
        dict: {anomalies, n_anomalies, anomaly_indices, method, threshold}
    """
    if len(values) < 4:
        return {"anomalies": [], "n_anomalies": 0, "anomaly_indices": [], "method": method}

    try:
        arr = np.array(values, dtype=float)
        anomalies = []

        if method == "zscore":
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            if std_val < 1e-10:
                return {"anomalies": [], "n_anomalies": 0, "anomaly_indices": [], "method": "zscore"}
            z_scores = (arr - mean_val) / std_val
            anomaly_mask = np.abs(z_scores) > threshold
            anomaly_indices = np.where(anomaly_mask)[0].tolist()
            for idx in anomaly_indices:
                anomalies.append({
                    "index": int(idx),
                    "value": float(arr[idx]),
                    "z_score": round(float(z_scores[idx]), 2),
                })

        elif method == "iqr":
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            anomaly_mask = (arr < lower) | (arr > upper)
            anomaly_indices = np.where(anomaly_mask)[0].tolist()
            for idx in anomaly_indices:
                anomalies.append({
                    "index": int(idx),
                    "value": float(arr[idx]),
                    "bounds": f"[{lower:.2f}, {upper:.2f}]",
                })

        return {
            "anomalies": anomalies,
            "n_anomalies": len(anomalies),
            "anomaly_indices": anomaly_indices,
            "method": method,
            "threshold": threshold,
        }
    except Exception as e:
        logger.error(f"异常检测失败: {e}")
        return {"anomalies": [], "n_anomalies": 0, "anomaly_indices": [], "method": method}


def detect_high_volatility_events(records, stability_key="anxiety_score", threshold=0.7):
    """
    检测情绪高波动事件

    计算相邻记录间稳定度的变化量，标记超过阈值的事件。

    参数：
        records (list): 历史记录列表（按时间排序）
        stability_key (str): 稳定度字段名
        threshold (float): 相邻变化阈值

    返回值：
        list: 高波动事件列表 [{index, timestamp, change, from_score, to_score}]
    """
    events = []
    if len(records) < 2:
        return events

    try:
        for i in range(1, len(records)):
            prev = float(records[i - 1].get(stability_key, 0))
            curr = float(records[i].get(stability_key, 0))
            change = curr - prev
            if abs(change) >= threshold:
                events.append({
                    "index": i,
                    "timestamp": records[i].get("timestamp", ""),
                    "change": round(change, 2),
                    "from_score": round(prev, 2),
                    "to_score": round(curr, 2),
                })
        return events
    except Exception as e:
        logger.error(f"波动事件检测失败: {e}")
        return events


# ===========================================================================
# 情绪分布比较
# ===========================================================================

def compare_emotion_distributions(dist1, dist2):
    """
    比较两个情绪分布的差异

    参数：
        dist1 (dict): 情绪分布1 {emotion: count}
        dist2 (dict): 情绪分布2 {emotion: count}

    返回值：
        dict: 包含分布差异指标的字典
    """
    try:
        all_emotions = set(list(dist1.keys()) + list(dist2.keys()))

        # 归一化为比例
        total1 = max(sum(dist1.values()), 1)
        total2 = max(sum(dist2.values()), 1)

        prop1 = {e: dist1.get(e, 0) / total1 for e in all_emotions}
        prop2 = {e: dist2.get(e, 0) / total2 for e in all_emotions}

        # 计算各情绪的比例差异
        diffs = {}
        for e in all_emotions:
            diffs[e] = round(prop2[e] - prop1[e], 4)

        # 总变异距离
        total_variation = sum(abs(diffs[e]) for e in all_emotions) / 2

        # 最大变化情绪
        max_change_emotion = max(diffs.items(), key=lambda x: abs(x[1]))

        return {
            "total_variation_distance": round(total_variation, 4),
            "emotion_differences": diffs,
            "max_change_emotion": max_change_emotion[0],
            "max_change_magnitude": max_change_emotion[1],
            "interpretation": "分布差异显著" if total_variation > 0.3 else "分布基本一致",
        }
    except Exception as e:
        logger.error(f"情绪分布比较失败: {e}")
        return {"total_variation_distance": 0, "emotion_differences": {}, "interpretation": "计算失败"}


# ===========================================================================
# 最小临床显著差异 (MCID) 计算
# ===========================================================================

def estimate_mcid(values, method="half_sd"):
    """
    估计最小临床显著差异 (MCID)

    使用分布法估计：MCID ≈ 0.5 × SD

    参数：
        values (list): 基准数据
        method (str): "half_sd" 或 "sem" (standard error of measurement)

    返回值：
        dict: {mcid, method, sd, note}
    """
    if len(values) < 2:
        return {"mcid": 0.0, "method": method, "note": "样本不足"}

    arr = np.array(values, dtype=float)
    sd = np.std(arr, ddof=1)

    if method == "half_sd":
        mcid = 0.5 * sd
    elif method == "sem":
        # SEM = SD × sqrt(1 - reliability)，假设 reliability = 0.8
        reliability = 0.8
        sem = sd * np.sqrt(1.0 - reliability)
        mcid = 1.96 * sem * np.sqrt(2)  # 显著变化阈值
    else:
        mcid = 0.5 * sd

    return {
        "mcid": round(mcid, 4),
        "method": method,
        "sd": round(float(sd), 4),
        "note": "基于分布法估计，待临床验证",
    }
