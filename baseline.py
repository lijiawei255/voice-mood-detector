# -*- coding: utf-8 -*-
"""
语音情绪识别系统 - 个人基线建模模块 (P2)

本模块提供个人基线校准功能，用于个体化评估：
1. 采集平静状态下的基线数据（3-5 条录音）
2. 计算个人基线指标（音高、音量、语速、情绪分布、稳定度）
3. 后续评估输出绝对分数 + 相对基线偏移
4. 个性化稳定度评分

基线指标说明：
- f0_baseline: 个人平均音高（Hz）
- rms_baseline: 个人平均音量范围
- speech_rate_baseline: 个人语速基线
- calm_prob_baseline: 平静状态下的情绪概率分布
- stability_baseline: 平均稳定度分数
- valence/arousal/dominance_baseline: VAD 维度基线

使用场景：
- 首次使用：采集 3-5 条平静语音建立基线
- 后续评估：同时输出绝对分数和相对基线偏移
- 个性化稳定度：基于基线偏差调整稳定度评分

⚠️ 基线需要定期更新（建议每月校准一次），以适应个体状态变化。

作者：Jiawei Li
许可证：GPL v3
"""

import os
import json
import logging
import numpy as np
from datetime import datetime

from app_paths import get_user_data_dir, safe_remove_file, is_safe_path

logger = logging.getLogger(__name__)

# 基线配置文件路径
BASELINE_FILE = "baseline.json"

# 最少需要的基线样本数
MIN_BASELINE_SAMPLES = 3
MAX_BASELINE_SAMPLES = 10


def get_baseline_path():
    """获取基线配置文件路径"""
    return os.path.join(get_user_data_dir(), BASELINE_FILE)


class PersonalBaseline:
    """
    个人基线管理器

    负责基线的建立、更新、查询和重置。
    基线数据保存在 portable_data/baseline.json 中。

    基线计算逻辑：
    - 音频特征取中位数（median），减少异常值影响
    - 情绪概率取均值（mean），保留分布特征
    - 稳定度取均值
    """

    def __init__(self):
        self.baseline_path = get_baseline_path()
        self.data = self._load()

    def _load(self):
        """从磁盘加载基线数据"""
        if not os.path.exists(self.baseline_path):
            return {"samples": [], "baseline": None, "established_at": None}
        try:
            with open(self.baseline_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {"samples": [], "baseline": None, "established_at": None}
            return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"基线文件加载失败: {e}")
            return {"samples": [], "baseline": None, "established_at": None}

    def _save(self):
        """保存基线数据到磁盘"""
        try:
            os.makedirs(os.path.dirname(self.baseline_path), exist_ok=True)
            with open(self.baseline_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"基线保存失败: {e}")
            return False

    def add_sample(self, result):
        """
        添加一条平静状态样本

        参数：
            result (dict): 情绪识别结果（需包含 acoustic_features 和 VAD 维度）

        返回值：
            bool: True 表示添加成功
        """
        if not isinstance(result, dict):
            return False

        try:
            sample = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "main_emotion": result.get("主要情绪", ""),
                "confidence": result.get("置信度", 0),
                "stability_score": result.get("情绪稳定度分数", 0),
                "valence_score": result.get("valence_score", 0),
                "arousal_score": result.get("arousal_score", 0),
                "dominance_score": result.get("dominance_score", 0),
                "negative_load": result.get("negative_load", 0),
                "emotional_uncertainty": result.get("emotional_uncertainty", 0),
                "probs_8": result.get("完整概率_8类", {}),
            }

            # 提取声学特征（如果有）
            acoustic = result.get("acoustic_features", {})
            if acoustic:
                sample["acoustic_features"] = {
                    "f0_mean": acoustic.get("f0_mean", 0),
                    "f0_std": acoustic.get("f0_std", 0),
                    "rms_mean": acoustic.get("rms_mean", 0),
                    "rms_std": acoustic.get("rms_std", 0),
                    "speech_rate": acoustic.get("speech_rate", 0),
                    "silence_ratio": acoustic.get("silence_ratio", 0),
                    "hnr_mean": acoustic.get("hnr_mean", 0),
                    "jitter_local": acoustic.get("jitter_local", 0),
                    "shimmer_local": acoustic.get("shimmer_local", 0),
                }

            self.data["samples"].append(sample)

            # 限制最大样本数
            if len(self.data["samples"]) > MAX_BASELINE_SAMPLES:
                self.data["samples"] = self.data["samples"][-MAX_BASELINE_SAMPLES:]

            # 如果有足够样本，计算基线
            if len(self.data["samples"]) >= MIN_BASELINE_SAMPLES:
                self._compute_baseline()

            return self._save()
        except Exception as e:
            logger.error(f"添加基线样本失败: {e}")
            return False

    def _compute_baseline(self):
        """从样本计算基线指标"""
        samples = self.data["samples"]
        if len(samples) < MIN_BASELINE_SAMPLES:
            return

        try:
            n = len(samples)

            # 稳定度统计
            stability_scores = [s.get("stability_score", 0) for s in samples]

            # VAD 维度统计
            valences = [s.get("valence_score", 0) for s in samples]
            arousals = [s.get("arousal_score", 0) for s in samples]
            dominances = [s.get("dominance_score", 0) for s in samples]
            neg_loads = [s.get("negative_load", 0) for s in samples]

            # 情绪概率平均分布
            prob_accum = {}
            prob_count = 0
            for s in samples:
                probs = s.get("probs_8", {})
                if probs:
                    prob_count += 1
                    for emo, p in probs.items():
                        prob_accum[emo] = prob_accum.get(emo, 0.0) + float(p)
            prob_baseline = {k: round(v / prob_count, 4) for k, v in prob_accum.items()} if prob_count > 0 else {}

            # 声学特征基线（取中位数）
            acoustic_baseline = {}
            acoustic_keys = ["f0_mean", "f0_std", "rms_mean", "rms_std",
                           "speech_rate", "silence_ratio", "hnr_mean",
                           "jitter_local", "shimmer_local"]
            for key in acoustic_keys:
                values = []
                for s in samples:
                    af = s.get("acoustic_features", {})
                    if af and key in af:
                        values.append(float(af[key]))
                if values:
                    acoustic_baseline[key] = round(float(np.median(values)), 4)

            self.data["baseline"] = {
                "stability_mean": round(float(np.mean(stability_scores)), 2),
                "stability_std": round(float(np.std(stability_scores, ddof=1)), 2),
                "valence_mean": round(float(np.mean(valences)), 4),
                "valence_std": round(float(np.std(valences, ddof=1)), 4),
                "arousal_mean": round(float(np.mean(arousals)), 4),
                "arousal_std": round(float(np.std(arousals, ddof=1)), 4),
                "dominance_mean": round(float(np.mean(dominances)), 4),
                "dominance_std": round(float(np.std(dominances, ddof=1)), 4),
                "negative_load_mean": round(float(np.mean(neg_loads)), 4),
                "calm_prob_distribution": prob_baseline,
                "acoustic_baseline": acoustic_baseline,
                "n_samples": n,
            }
            self.data["established_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        except Exception as e:
            logger.error(f"基线计算失败: {e}")

    def get_baseline(self):
        """获取当前基线数据"""
        return self.data.get("baseline", None)

    def is_established(self):
        """基线是否已建立"""
        return self.data.get("baseline") is not None and len(self.data.get("samples", [])) >= MIN_BASELINE_SAMPLES

    def get_sample_count(self):
        """获取已有基线样本数"""
        return len(self.data.get("samples", []))

    def compute_deviation(self, result):
        """
        计算当前评估结果相对基线的偏移

        参数：
            result (dict): 当前情绪识别结果

        返回值：
            dict: 包含绝对值和偏移量的字典
        """
        baseline = self.get_baseline()
        if not baseline:
            return {"available": False, "note": "基线尚未建立"}

        try:
            deviations = {}

            # 稳定度偏移
            current_stability = result.get("情绪稳定度分数", 0)
            base_stability = baseline.get("stability_mean", 0)
            base_std = max(baseline.get("stability_std", 1), 0.5)
            deviations["stability_deviation"] = round(current_stability - base_stability, 2)
            deviations["stability_deviation_z"] = round((current_stability - base_stability) / base_std, 2)

            # VAD 维度偏移
            for dim in ["valence", "arousal", "dominance"]:
                current = result.get(f"{dim}_score", 0)
                base_mean = baseline.get(f"{dim}_mean", 0)
                base_std = max(baseline.get(f"{dim}_std", 0.1), 0.05)
                deviations[f"{dim}_deviation"] = round(current - base_mean, 4)
                deviations[f"{dim}_deviation_z"] = round((current - base_mean) / base_std, 2)

            # 个性化稳定度评分（基于基线偏移调整）
            # 如果当前稳定度显著偏离基线（>1.5 std），标记为异常波动
            personalized_stability = self._compute_personalized_stability(
                current_stability, base_stability, base_std, deviations
            )

            return {
                "available": True,
                "baseline_established_at": self.data.get("established_at"),
                "baseline_n_samples": baseline.get("n_samples", 0),
                "baseline_stability_mean": base_stability,
                "deviations": deviations,
                "personalized_stability_score": personalized_stability["score"],
                "personalized_stability_level": personalized_stability["level"],
                "is_significant_deviation": personalized_stability["is_significant"],
                "summary": self._generate_deviation_summary(deviations, personalized_stability),
            }

        except Exception as e:
            logger.error(f"基线偏差计算失败: {e}")
            return {"available": False, "note": str(e)}

    def _compute_personalized_stability(self, current, baseline_mean, baseline_std, deviations):
        """
        计算个性化稳定度评分

        基于基线偏移进行调整：显著偏离基线时降低稳定性评分。
        """
        z = abs(deviations.get("stability_deviation_z", 0))

        if z < 0.5:
            level = "与基线一致"
            score = max(0, current - 0.5)
            significant = False
        elif z < 1.0:
            level = "轻微偏离基线"
            score = current
            significant = False
        elif z < 2.0:
            level = "中度偏离基线"
            score = min(10, current + 1.0)
            significant = True
        else:
            level = "显著偏离基线"
            score = min(10, current + 2.0)
            significant = True

        return {"score": round(score, 2), "level": level, "is_significant": significant}

    def _generate_deviation_summary(self, deviations, personalized):
        """生成基线偏差摘要"""
        parts = []
        stab_z = deviations.get("stability_deviation_z", 0)
        if abs(stab_z) > 1.0:
            direction = "升高" if stab_z > 0 else "降低"
            parts.append(f"情绪稳定度较基线{direction}（z={stab_z:+.1f}）")

        valence_z = deviations.get("valence_deviation_z", 0)
        if abs(valence_z) > 1.0:
            direction = "偏正向" if valence_z > 0 else "偏负向"
            parts.append(f"效价{direction}（z={valence_z:+.1f}）")

        arousal_z = deviations.get("arousal_deviation_z", 0)
        if abs(arousal_z) > 1.0:
            direction = "升高" if arousal_z > 0 else "降低"
            parts.append(f"唤醒度{direction}（z={arousal_z:+.1f}）")

        if not parts:
            parts.append("当前状态与个人基线一致")

        return "；".join(parts)

    def reset(self):
        """重置基线（清空所有样本和基线数据）"""
        self.data = {"samples": [], "baseline": None, "established_at": None}
        return self._save()

    def remove_last_sample(self):
        """删除最后一个基线样本"""
        if self.data["samples"]:
            self.data["samples"].pop()
            if len(self.data["samples"]) >= MIN_BASELINE_SAMPLES:
                self._compute_baseline()
            else:
                self.data["baseline"] = None
                self.data["established_at"] = None
            return self._save()
        return False
