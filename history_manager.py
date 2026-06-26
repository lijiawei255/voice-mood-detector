# -*- coding: utf-8 -*-
"""
语音情绪识别系统 - 历史记录管理模块

本模块负责管理情绪识别的历史记录，包括记录的增删改查、
持久化存储、以及自动清理过期记录等功能。

主要功能：
1. 历史记录的添加、查询、删除
2. JSON 文件持久化存储（原子写入，防止损坏）
3. 自动限制最大记录数（防止数据无限增长）
4. 记录数据清洗与校验（防止恶意数据注入）
5. 关联音频文件的安全删除

数据结构：
每条历史记录包含以下字段：
- id: 唯一标识符（时间戳 + 随机后缀）
- timestamp: 检测时间（YYYY-MM-DD HH:MM:SS）
- audio_file: 关联的音频文件路径
- main_emotion: 主要情绪类型
- anxiety_score: 情绪稳定度分数（0-10）
- emotion_level: 情绪状态等级
- confidence: 识别置信度（0-1）
- suggestion: 调节建议文本

作者：Jiawei Li
许可证：GPL v3
"""

import os
import json
import uuid
from datetime import datetime
import logging

from app_paths import get_history_path, get_recordings_dir, safe_remove_file, is_safe_path

logger = logging.getLogger(__name__)


def _safe_float(value, min_val, max_val):
    """安全转换浮点数到指定范围"""
    try:
        v = float(value)
        return round(max(min_val, min(max_val, v)), 6)
    except (TypeError, ValueError):
        return min_val


class HistoryManager:
    """
    历史记录管理器类

    负责历史记录的生命周期管理，包括：
    - 加载和保存历史记录到 JSON 文件
    - 添加新记录（自动清理超过上限的旧记录）
    - 查询记录（支持按时间倒序排列）
    - 删除记录（同时删除关联的音频文件）

    设计特点：
    - 原子写入：使用临时文件 + 重命名的方式，防止写入中断导致文件损坏
    - 数据校验：所有输入数据都会经过清洗和验证
    - 自动清理：超过 MAX_RECORDS 条时自动删除最旧的记录
    - 安全删除：音频文件删除前会进行路径安全校验
    """

    MAX_RECORDS = 200           # 最大记录数
    MIN_RECORDS_TO_KEEP = 50    # 最少保留记录数
    AUTO_CLEAN_THRESHOLD = 50   # 自动清理触发阈值
    AUTO_CLEAN_COUNT = 10       # 每次自动清理的记录数

    def __init__(self, cleanup_callback=None):
        """
        初始化历史记录管理器

        初始化时会自动从磁盘加载已有的历史记录。
        如果历史文件不存在或损坏，会初始化为空记录。

        参数：
            cleanup_callback (callable, 可选): 自动清理前的回调函数，
                签名为 callback(count)，用于通知 GUI 显示提醒
        """
        self.history_path = get_history_path()
        self.records = []
        self.cleanup_callback = cleanup_callback
        self._ensure_directory()
        self.load()

    def _ensure_directory(self):
        """
        确保历史记录文件所在目录存在（内部方法）
        """
        try:
            directory = os.path.dirname(self.history_path)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
        except Exception as e:
            logger.error(f"创建历史目录失败: {str(e)}")

    def _sanitize_record(self, record):
        """
        清洗和校验记录数据（内部方法）

        对输入的记录数据进行验证和清理，
        防止无效或恶意数据进入系统。

        校验规则：
        - id: 转为字符串，确保存在
        - timestamp: 时间字符串格式
        - audio_file: 字符串，防止路径注入
        - main_emotion: 限制最长20字符
        - anxiety_score: 限制在 0-10 范围
        - emotion_level: 限制最长10字符
        - confidence: 限制在 0-1 范围
        - suggestion: 限制最长500字符
        - compound_emotion: 复合情绪名称，限制最长20字符
        - emotion_summary: 情绪分析摘要，限制最长500字符

        参数：
            record (dict): 待清洗的记录数据

        返回值：
            dict 或 None: 清洗后的记录数据，如果输入无效则返回 None
        """
        if not isinstance(record, dict):
            return None
        sanitized = {}
        sanitized['id'] = str(record.get('id', uuid.uuid4().hex))
        sanitized['timestamp'] = str(record.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        sanitized['audio_file'] = str(record.get('audio_file', ''))
        sanitized['main_emotion'] = str(record.get('main_emotion', '未知'))[:20]
        try:
            sanitized['anxiety_score'] = float(record.get('anxiety_score', 0.0))
            sanitized['anxiety_score'] = max(0.0, min(10.0, sanitized['anxiety_score']))
        except (TypeError, ValueError):
            sanitized['anxiety_score'] = 0.0
        sanitized['emotion_level'] = str(record.get('emotion_level', '未知'))[:10]
        try:
            sanitized['confidence'] = float(record.get('confidence', 0.0))
            sanitized['confidence'] = max(0.0, min(1.0, sanitized['confidence']))
        except (TypeError, ValueError):
            sanitized['confidence'] = 0.0
        sanitized['suggestion'] = str(record.get('suggestion', ''))[:500]
        # 新增字段（向后兼容：旧记录可能没有这些字段）
        sanitized['compound_emotion'] = str(record.get('compound_emotion', ''))[:20]
        sanitized['emotion_summary'] = str(record.get('emotion_summary', ''))[:500]
        # P0 新增字段：VAD 维度指标
        sanitized['valence_score'] = _safe_float(record.get('valence_score'), -1.0, 1.0)
        sanitized['arousal_score'] = _safe_float(record.get('arousal_score'), 0.0, 1.0)
        sanitized['dominance_score'] = _safe_float(record.get('dominance_score'), 0.0, 1.0)
        sanitized['negative_load'] = _safe_float(record.get('negative_load'), 0.0, 1.0)
        sanitized['emotional_uncertainty'] = _safe_float(record.get('emotional_uncertainty'), 0.0, 1.0)
        # P0 新增字段：原始输出与完整概率（用于科研复算）
        sanitized['raw_model_output'] = record.get('raw_model_output', {}) if isinstance(record.get('raw_model_output'), dict) else {}
        sanitized['probs_8'] = record.get('probs_8', {}) if isinstance(record.get('probs_8'), dict) else {}
        sanitized['probs_7'] = record.get('probs_7', {}) if isinstance(record.get('probs_7'), dict) else {}
        # P0 新增字段：稳定度分项因子
        sanitized['stability_factors'] = record.get('stability_factors', {}) if isinstance(record.get('stability_factors'), dict) else {}
        # P0 新增字段：音频质量
        sanitized['audio_quality'] = record.get('audio_quality', {}) if isinstance(record.get('audio_quality'), dict) else {}
        # P0 新增字段：实验元数据（可复现性基础）
        sanitized['model_name'] = str(record.get('model_name', ''))[:50]
        sanitized['algorithm_version'] = str(record.get('algorithm_version', ''))[:20]
        sanitized['app_version'] = str(record.get('app_version', ''))[:20]
        sanitized['is_research_mode'] = bool(record.get('is_research_mode', False))
        sanitized['assessment_reliability'] = str(record.get('assessment_reliability', ''))[:20]
        # P0 新增字段：估计标注说明
        sanitized['estimation_note'] = str(record.get('estimation_note', ''))[:500]
        return sanitized

    def add_record(self, result):
        """
        添加一条新的历史记录

        将情绪识别结果保存为历史记录。
        如果记录总数超过 MAX_RECORDS，会自动删除最旧的记录
        及其关联的音频文件。

        参数：
            result (dict): 情绪识别结果字典（来自 EmotionRecognizer.predict()）

        返回值：
            bool: True 表示添加成功，False 表示失败
        """
        try:
            # 从识别结果中提取字段
            main_emotion = str(result.get("主要情绪", result.get("main_emotion", "未知")))
            stability_score = result.get("情绪稳定度分数", result.get("anxiety_score", 0.0))
            emotion_level = str(result.get("情绪状态等级", result.get("emotion_level", "未知")))
            confidence = result.get("置信度", result.get("confidence", 0.0))

            # 数值类型转换和范围限制
            try:
                score_f = float(stability_score)
            except (TypeError, ValueError):
                score_f = 0.0

            try:
                conf_f = float(confidence)
                if conf_f > 1.0:
                    conf_f = conf_f / 100.0
            except (TypeError, ValueError):
                conf_f = 0.0

            # 构建记录对象（含P0新增科研元数据字段）
            record = {
                "id": datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "audio_file": str(result.get("audio_file", "")),
                "main_emotion": main_emotion,
                "anxiety_score": score_f,
                "emotion_level": emotion_level,
                "confidence": conf_f,
                "suggestion": str(result.get("suggestion_text", result.get("调节建议", "")))[:500],
                "compound_emotion": str(result.get("复合情绪", ""))[:20],
                "emotion_summary": str(result.get("情绪分析摘要", ""))[:500],
                # P0 新增：VAD 维度指标
                "valence_score": result.get("valence_score", 0.0),
                "arousal_score": result.get("arousal_score", 0.0),
                "dominance_score": result.get("dominance_score", 0.0),
                "negative_load": result.get("negative_load", 0.0),
                "emotional_uncertainty": result.get("emotional_uncertainty", 0.0),
                # P0 新增：原始输出与完整概率
                "raw_model_output": result.get("原始模型输出", {}),
                "probs_8": result.get("完整概率_8类", {}),
                "probs_7": result.get("所有情绪概率", {}),
                # P0 新增：稳定度分项
                "stability_factors": result.get("稳定度分项", {}),
                # P0 新增：实验元数据
                "model_name": result.get("model_name", ""),
                "algorithm_version": result.get("algorithm_version", "2.0.0-p0"),
                "app_version": result.get("app_version", "2.0.0"),
                "is_research_mode": bool(result.get("is_research_mode", False)),
                "assessment_reliability": str(result.get("assessment_reliability", ""))[:20],
                "estimation_note": str(result.get("estimation_note", ""))[:500],
            }
            # 数据清洗
            record = self._sanitize_record(record)
            if record is None:
                logger.warning("无效的记录数据，跳过添加")
                return False

            self.records.append(record)

            # 科研模式下禁用自动清理（保证长期追踪数据完整性）
            is_research = bool(result.get("is_research_mode", False))
            if not is_research:
                # 达到自动清理阈值时，通知并删除最早的记录
                if len(self.records) >= self.AUTO_CLEAN_THRESHOLD:
                    if self.cleanup_callback:
                        try:
                            self.cleanup_callback(self.AUTO_CLEAN_COUNT)
                        except Exception:
                            pass
                    # 删除最早的 AUTO_CLEAN_COUNT 条记录
                    to_delete = self.records[:self.AUTO_CLEAN_COUNT]
                    for old_rec in to_delete:
                        self._delete_audio_file(old_rec.get("audio_file", ""))
                    self.records = self.records[self.AUTO_CLEAN_COUNT:]
                    logger.info(f"自动清理: 删除最早的 {self.AUTO_CLEAN_COUNT} 条历史记录")

                # 超过最大记录数时，清理最旧的记录
                if len(self.records) > self.MAX_RECORDS:
                    oldest = self.records[:-self.MAX_RECORDS]
                    for old_rec in oldest:
                        self._delete_audio_file(old_rec.get("audio_file", ""))
                    self.records = self.records[-self.MAX_RECORDS:]

            self.save()
            return True
        except Exception as e:
            logger.error(f"添加记录失败: {str(e)}", exc_info=True)
            return False

    def _delete_audio_file(self, filepath):
        """
        安全删除关联的音频文件（内部方法）

        删除前会进行路径安全校验，确保文件在用户数据目录内，
        防止路径遍历攻击导致误删系统文件。

        参数：
            filepath (str): 音频文件路径
        """
        if not filepath:
            return
        try:
            if not is_safe_path(filepath):
                logger.warning(f"拒绝删除非数据目录文件: {filepath}")
                return
            safe_remove_file(filepath)
        except Exception as e:
            logger.warning(f"删除音频文件失败 {filepath}: {str(e)}")

    def get_records(self):
        """
        获取所有历史记录（按时间倒序排列）

        返回值：
            list: 历史记录列表，按时间从新到旧排序
        """
        try:
            # 重新清洗所有记录，确保数据有效性
            valid_records = []
            for r in self.records:
                sanitized = self._sanitize_record(r)
                if sanitized:
                    valid_records.append(sanitized)
            self.records = valid_records
            # 按时间倒序排列（最新的在前）
            return sorted(self.records, key=lambda x: x.get("timestamp", ""), reverse=True)
        except Exception as e:
            logger.error(f"获取记录失败: {str(e)}")
            return []

    def get_record_by_id(self, record_id):
        """
        根据 ID 获取单条记录

        参数：
            record_id (str): 记录 ID

        返回值：
            dict 或 None: 找到的记录，未找到返回 None
        """
        if not record_id or not isinstance(record_id, str):
            return None
        for rec in self.records:
            if rec.get("id") == record_id:
                return self._sanitize_record(rec)
        return None

    def delete_record(self, record_id):
        """
        删除单条历史记录

        同时删除关联的音频文件。

        参数：
            record_id (str): 要删除的记录 ID

        返回值：
            bool: True 表示删除成功，False 表示失败或未找到
        """
        if not record_id or not isinstance(record_id, str):
            return False
        try:
            record_to_delete = None
            for rec in self.records:
                if rec.get("id") == record_id:
                    record_to_delete = rec
                    break
            if record_to_delete:
                # 删除关联的音频文件
                self._delete_audio_file(record_to_delete.get("audio_file", ""))
                # 从列表中移除
                self.records = [r for r in self.records if r.get("id") != record_id]
                self.save()
                return True
            return False
        except Exception as e:
            logger.error(f"删除记录失败: {str(e)}")
            return False

    def delete_records(self, record_ids):
        """
        批量删除历史记录

        参数：
            record_ids (list/tuple/set): 要删除的记录 ID 列表

        返回值：
            int: 成功删除的记录数量
        """
        if not record_ids or not isinstance(record_ids, (list, tuple, set)):
            return 0
        deleted_count = 0
        try:
            ids_to_delete = set(str(rid) for rid in record_ids if rid)
            records_to_keep = []
            for rec in self.records:
                if rec.get("id") in ids_to_delete:
                    # 删除关联的音频文件
                    self._delete_audio_file(rec.get("audio_file", ""))
                    deleted_count += 1
                else:
                    records_to_keep.append(rec)
            self.records = records_to_keep
            if deleted_count > 0:
                self.save()
            return deleted_count
        except Exception as e:
            logger.error(f"批量删除记录失败: {str(e)}")
            return deleted_count

    def delete_all_records(self):
        """
        删除所有历史记录

        同时删除所有关联的音频文件。

        返回值：
            bool: True 表示删除成功，False 表示失败
        """
        try:
            for rec in self.records:
                self._delete_audio_file(rec.get("audio_file", ""))
            self.records = []
            self.save()
            return True
        except Exception as e:
            logger.error(f"清空记录失败: {str(e)}")
            return False

    def save(self):
        """
        保存历史记录到文件（原子写入）

        使用"临时文件 + 重命名"的原子写入策略：
        1. 先写入到 .tmp 临时文件
        2. 将旧文件备份为 .bak
        3. 将临时文件重命名为正式文件

        这样即使写入过程中程序崩溃，也不会丢失历史数据。

        返回值：
            bool: True 表示保存成功，False 表示失败
        """
        try:
            self._ensure_directory()
            temp_path = self.history_path + '.tmp'
            # 写入临时文件
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.records, f, ensure_ascii=False, indent=2)
            # 备份旧文件
            if os.path.exists(self.history_path):
                backup_path = self.history_path + '.bak'
                try:
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    os.rename(self.history_path, backup_path)
                except OSError:
                    pass
            # 原子替换
            os.rename(temp_path, self.history_path)
            return True
        except Exception as e:
            logger.error(f"保存历史记录失败: {str(e)}", exc_info=True)
            # 清理临时文件
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            return False

    def load(self):
        """
        从文件加载历史记录

        如果文件不存在，初始化为空列表。
        如果文件损坏，会将损坏的文件重命名为 .corrupted 并初始化为空。
        """
        if not os.path.exists(self.history_path):
            self.records = []
            return
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                valid_records = []
                for item in data:
                    sanitized = self._sanitize_record(item)
                    if sanitized:
                        valid_records.append(sanitized)
                self.records = valid_records
            else:
                self.records = []
                logger.warning("历史记录格式无效，已重置")
        except (json.JSONDecodeError, IOError, UnicodeDecodeError) as e:
            logger.error(f"加载历史记录失败: {str(e)}")
            # 将损坏的文件备份
            try:
                corrupted = self.history_path + '.corrupted'
                if os.path.exists(corrupted):
                    os.remove(corrupted)
                os.rename(self.history_path, corrupted)
            except OSError:
                pass
            self.records = []

    def get_record_count(self):
        """
        获取历史记录总数

        返回值：
            int: 记录数量
        """
        return len(self.records)
