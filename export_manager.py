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
    "id", "timestamp",
    # 主要结果
    "main_emotion", "confidence", "stability_score", "stability_level",
    # 稳定度分项
    "negative_weight_score", "entropy_score", "extremity_score",
    # VAD 维度
    "valence_score", "arousal_score", "dominance_score",
    "negative_load", "emotional_uncertainty",
    # 复合情绪
    "compound_emotion",
    # 音频质量
    "audio_quality_score", "audio_quality_label", "speech_ratio",
    "rms_mean", "noise_level", "clipping_ratio",
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

    return {
        "id": record.get('id', ''),
        "timestamp": record.get('timestamp', ''),
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
        "compound_emotion": record.get('compound_emotion', ''),
        "audio_quality_score": audio_q.get('quality_score', ''),
        "audio_quality_label": audio_q.get('quality_label', ''),
        "speech_ratio": audio_q.get('speech_ratio', ''),
        "rms_mean": audio_q.get('rms_mean', ''),
        "noise_level": audio_q.get('noise_level', ''),
        "clipping_ratio": audio_q.get('clipping_ratio', ''),
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
