# -*- coding: utf-8 -*-
"""
语音情绪识别系统 - 数据导出模块 (P1)

本模块提供研究数据的标准化导出功能：
1. CSV 导出（SPSS / Excel 兼容，UTF-8 BOM）
2. JSON 研究数据集导出（完整结构化记录）
3. 会话报告导出

导出字段包含完整的科研元数据，确保数据可追溯和可复算。

作者：Jiawei Li
许可证：GPL v3
"""

import os
import json
import csv
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# CSV 导出字段列表（按逻辑分组排列）
CSV_FIELDS = [
    # 基本信息
    "id", "timestamp", "recording_duration",
    # 主要结果
    "main_emotion", "confidence", "stability_score", "stability_level",
    # 稳定度分项
    "negative_weight_score", "entropy_score", "extremity_score",
    # VAD 维度
    "valence_score", "arousal_score", "dominance_score",
    "negative_load", "emotional_uncertainty",
    "estimation_note",
    # 复合情绪
    "compound_emotion", "compound_emotion_description",
    # 音频质量（完整）
    "audio_quality_score", "audio_quality_label", "speech_ratio",
    "rms_mean", "noise_level", "clipping_ratio",
    "audio_quality_issues",
    # 元数据
    "model_name", "algorithm_version", "app_version",
    "is_research_mode", "assessment_reliability",
    # 调节建议
    "suggestion",
]


def export_records_csv(records, output_path):
    """
    将历史记录导出为 CSV 文件（SPSS/Excel 兼容）

    使用 UTF-8 BOM 编码确保中文在 Excel 中正确显示。

    参数：
        records (list): 历史记录列表（来自 HistoryManager.get_records()）
        output_path (str): 输出 CSV 文件路径

    返回值：
        bool: True 表示导出成功
    """
    if not records:
        logger.warning("没有记录可导出")
        return False

    try:
        with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(CSV_FIELDS)
            # 写入数据行
            for rec in records:
                row = _build_csv_row(rec)
                writer.writerow([row.get(field, "") for field in CSV_FIELDS])

        logger.info(f"CSV 导出完成: {output_path} ({len(records)} 条记录)")
        return True
    except Exception as e:
        logger.error(f"CSV 导出失败: {e}")
        return False


def _build_csv_row(record):
    """从记录构建 CSV 行数据"""
    factors = record.get('stability_factors', {})
    if not isinstance(factors, dict):
        factors = {}
    audio_q = record.get('audio_quality', {})
    if not isinstance(audio_q, dict):
        audio_q = {}

    # 录音时长：从 audio_quality 中获取，或从 audio_file 推断
    duration = audio_q.get('duration', '')
    if not duration:
        duration = record.get('duration', '')

    # 音频质量问题列表
    issues = audio_q.get('issues', [])
    issues_str = '; '.join(issues) if issues else ''

    return {
        "id": record.get('id', ''),
        "timestamp": record.get('timestamp', ''),
        "recording_duration": duration,
        "main_emotion": record.get('main_emotion', ''),
        "confidence": record.get('confidence', 0),
        "stability_score": record.get('anxiety_score', 0),
        "stability_level": record.get('emotion_level', ''),
        "negative_weight_score": factors.get('negative_weight_score', ''),
        "entropy_score": factors.get('entropy_score', ''),
        "extremity_score": factors.get('extremity_score', ''),
        "valence_score": record.get('valence_score', ''),
        "arousal_score": record.get('arousal_score', ''),
        "dominance_score": record.get('dominance_score', ''),
        "negative_load": record.get('negative_load', ''),
        "emotional_uncertainty": record.get('emotional_uncertainty', ''),
        "estimation_note": record.get('estimation_note', ''),
        "compound_emotion": record.get('compound_emotion', ''),
        "compound_emotion_description": (record.get('compound_emotion_detail', {}) or {}).get('desc', ''),
        "audio_quality_score": audio_q.get('quality_score', ''),
        "audio_quality_label": audio_q.get('quality_label', ''),
        "speech_ratio": audio_q.get('speech_ratio', ''),
        "rms_mean": audio_q.get('rms_mean', ''),
        "noise_level": audio_q.get('noise_level', ''),
        "clipping_ratio": audio_q.get('clipping_ratio', ''),
        "audio_quality_issues": issues_str,
        "model_name": record.get('model_name', ''),
        "algorithm_version": record.get('algorithm_version', ''),
        "app_version": record.get('app_version', ''),
        "is_research_mode": record.get('is_research_mode', ''),
        "assessment_reliability": record.get('assessment_reliability', ''),
        "suggestion": record.get('suggestion', ''),
    }


def export_research_dataset(records, output_path):
    """
    导出完整的研究数据集（JSON 格式）

    包含所有原始模型输出、完整概率分布、分项分数等，
    确保科研复算和复审的完整可追溯性。

    参数：
        records (list): 历史记录列表
        output_path (str): 输出 JSON 文件路径

    返回值：
        bool: True 表示导出成功
    """
    if not records:
        logger.warning("没有记录可导出")
        return False

    try:
        dataset = {
            "export_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "record_count": len(records),
            "schema_version": "2.0.0",
            "records": records
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        logger.info(f"研究数据集导出完成: {output_path} ({len(records)} 条)")
        return True
    except Exception as e:
        logger.error(f"研究数据集导出失败: {e}")
        return False


def export_session_report(results, output_path):
    """
    导出单次会话的结构化报告（JSON）

    包含一次完整评估的所有信息，适合作为研究文档附件。

    参数：
        results (dict): 预测结果字典
        output_path (str): 输出路径

    返回值：
        bool: True 表示导出成功
    """
    if not results:
        return False

    try:
        report = {
            "report_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "report_version": "2.0.0",
            "assessment": {
                "primary_emotion": results.get("主要情绪", ""),
                "confidence": results.get("置信度", 0),
                "stability_score": results.get("情绪稳定度分数", 0),
                "stability_level": results.get("情绪状态等级", ""),
                "stability_factors": results.get("稳定度分项", {}),
            },
            "vad_dimensions": {
                "valence": results.get("valence_score", 0),
                "arousal": results.get("arousal_score", 0),
                "dominance": results.get("dominance_score", 0),
                "negative_load": results.get("negative_load", 0),
                "emotional_uncertainty": results.get("emotional_uncertainty", 0),
                "estimation_note": results.get("estimation_note", ""),
            },
            "compound_emotion": results.get("复合情绪详情", None),
            "probabilities_8class": results.get("完整概率_8类", {}),
            "probabilities_7class": results.get("所有情绪概率", {}),
            "audio_quality": results.get("audio_quality", {}),
            "reliability": results.get("assessment_reliability", ""),
            "metadata": {
                "model_name": results.get("model_name", ""),
                "algorithm_version": results.get("algorithm_version", ""),
                "app_version": results.get("app_version", ""),
                "is_research_mode": results.get("is_research_mode", False),
            },
            "suggestion": results.get("调节建议", ""),
            "summary": results.get("情绪分析摘要", ""),
        }

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"会话报告导出完成: {output_path}")
        return True
    except Exception as e:
        logger.error(f"会话报告导出失败: {e}")
        return False


# ===========================================================================
# P2 新增：标准化评估报告生成
# ===========================================================================

def generate_standardized_report(results, acoustic_features=None, baseline_deviation=None):
    """
    生成标准化心理评估报告（P2）

    报告结构符合科研论文撰写要求，包含以下部分：
    1. 基础情绪评估
    2. VAD 维度指标
    3. 心理状态指标
    4. 声音质量指标
    5. 可靠性指标
    6. 基线偏差（如可用）
    7. 元数据

    参数：
        results (dict): 情绪识别结果
        acoustic_features (dict): 声学特征（可选）
        baseline_deviation (dict): 基线偏移数据（可选）

    返回值：
        dict: 标准化评估报告
    """
    if not results:
        return {"error": "无结果数据"}

    report = {
        "report_type": "标准化语音心理状态评估报告",
        "report_version": "2.0.0",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sections": {}
    }

    # ---- 1. 基础情绪评估 ----
    probs_7 = results.get("所有情绪概率", {})
    probs_8 = results.get("完整概率_8类", {})

    report["sections"]["basic_emotion"] = {
        "title": "基础情绪评估",
        "primary_emotion": results.get("主要情绪", ""),
        "confidence": results.get("置信度", 0),
        "probabilities_8class": probs_8,
        "display_probabilities_7class": probs_7,
        "mixed_emotions": [
            {"emotion": e, "probability": p}
            for e, p in results.get("混合情绪", [])
        ],
    }

    # 复合情绪
    compound_detail = results.get("复合情绪详情")
    if compound_detail and isinstance(compound_detail, dict):
        report["sections"]["compound_emotion"] = {
            "title": "复合情绪分析",
            "name": compound_detail.get("name", ""),
            "confidence": compound_detail.get("confidence", 0),
            "components": compound_detail.get("components", {}),
            "description": compound_detail.get("desc", ""),
            "interpretation": compound_detail.get("interpretation", ""),
        }

    # ---- 2. VAD 维度指标 ----
    report["sections"]["vad_dimensions"] = {
        "title": "情感维度指标（效价-唤醒度-掌控感）",
        "valence": {
            "score": results.get("valence_score", 0),
            "interpretation": _interpret_valence(results.get("valence_score", 0)),
        },
        "arousal": {
            "score": results.get("arousal_score", 0),
            "interpretation": _interpret_arousal(results.get("arousal_score", 0)),
        },
        "dominance": {
            "score": results.get("dominance_score", 0),
            "interpretation": _interpret_dominance(results.get("dominance_score", 0)),
        },
        "negative_load": results.get("negative_load", 0),
        "emotional_uncertainty": results.get("emotional_uncertainty", 0),
        "estimation_note": results.get("estimation_note", ""),
    }

    # ---- 3. 心理状态指标 ----
    stability_factors = results.get("稳定度分项", {})

    psych_section = {
        "title": "心理状态指标",
        "stability": {
            "score": results.get("情绪稳定度分数", 0),
            "level": results.get("情绪状态等级", ""),
            "factors": {
                "negative_weight": stability_factors.get("negative_weight_score", 0),
                "entropy": stability_factors.get("entropy_score", 0),
                "extremity": stability_factors.get("extremity_score", 0),
            },
            "factor_weights_source": stability_factors.get("factor_weights_source", ""),
        },
    }

    # 添加心理状态映射指标（如果可用）
    if acoustic_features:
        from audio_features import compute_psychological_indicators
        vad_dims = {
            "valence_score": results.get("valence_score", 0),
            "arousal_score": results.get("arousal_score", 0),
            "dominance_score": results.get("dominance_score", 0),
            "negative_load": results.get("negative_load", 0),
        }
        indicators = compute_psychological_indicators(acoustic_features, vad_dims)
        psych_section["psychological_indicators"] = indicators

    report["sections"]["psychological_state"] = psych_section

    # ---- 4. 声音质量指标 ----
    audio_q = results.get("audio_quality", {})
    if audio_q and isinstance(audio_q, dict):
        report["sections"]["audio_quality"] = {
            "title": "声音质量指标",
            "quality_score": audio_q.get("quality_score", 0),
            "quality_label": audio_q.get("quality_label", ""),
            "speech_ratio": audio_q.get("speech_ratio", 0),
            "noise_level": audio_q.get("noise_level", 0),
            "rms_mean": audio_q.get("rms_mean", 0),
            "clipping_ratio": audio_q.get("clipping_ratio", 0),
            "issues": audio_q.get("issues", []),
            "recommendations": audio_q.get("recommendations", []),
        }

    # ---- 5. 可靠性指标 ----
    report["sections"]["reliability"] = {
        "title": "可靠性评估",
        "assessment_reliability": results.get("assessment_reliability", "未评估"),
        "confidence": results.get("置信度", 0),
        "estimation_note": results.get("estimation_note", ""),
    }

    # ---- 6. 基线偏差 ----
    if baseline_deviation and baseline_deviation.get("available"):
        report["sections"]["baseline_comparison"] = {
            "title": "个人基线对比",
            "available": True,
            "baseline_established_at": baseline_deviation.get("baseline_established_at"),
            "baseline_n_samples": baseline_deviation.get("baseline_n_samples"),
            "baseline_stability_mean": baseline_deviation.get("baseline_stability_mean"),
            "deviations": baseline_deviation.get("deviations", {}),
            "personalized_stability_score": baseline_deviation.get("personalized_stability_score"),
            "personalized_stability_level": baseline_deviation.get("personalized_stability_level"),
            "is_significant_deviation": baseline_deviation.get("is_significant_deviation"),
            "summary": baseline_deviation.get("summary", ""),
        }

    # ---- 7. 元数据 ----
    report["metadata"] = {
        "model_name": results.get("model_name", ""),
        "algorithm_version": results.get("algorithm_version", ""),
        "app_version": results.get("app_version", ""),
        "is_research_mode": results.get("is_research_mode", False),
        "is_dual_model": results.get("is_dual_model", False),
    }

    if results.get("is_dual_model"):
        report["metadata"]["primary_model"] = results.get("primary_model", "")
        report["metadata"]["secondary_model"] = results.get("secondary_model", "")
        report["metadata"]["model_agreement"] = results.get("model_agreement", 0)

    # ---- 建议 ----
    report["suggestion"] = results.get("调节建议", "")
    report["summary"] = results.get("情绪分析摘要", "")

    return report


def _interpret_valence(v):
    """解释效价分数"""
    if v > 0.3:
        return "正性情绪主导"
    elif v < -0.3:
        return "负性情绪主导"
    else:
        return "中性"

def _interpret_arousal(a):
    """解释唤醒度"""
    if a > 0.7:
        return "高激活状态"
    elif a > 0.3:
        return "中等激活"
    else:
        return "低激活状态"

def _interpret_dominance(d):
    """解释掌控感"""
    if d > 0.7:
        return "高掌控感"
    elif d > 0.3:
        return "中等掌控感"
    else:
        return "低掌控感"
