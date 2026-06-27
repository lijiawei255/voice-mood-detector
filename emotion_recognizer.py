# -*- coding: utf-8 -*-
"""
语音情绪识别系统 - 情绪识别核心模块

本模块是整个系统的核心，负责加载 AI 模型并进行语音情绪识别。
使用 ModelScope / FunASR 加载 emotion2vec+ 系列预训练模型，
对音频文件进行推理，识别其中的情绪状态。

主要功能：
1. 单例模式的情绪识别器（全局唯一实例，避免重复加载模型）
2. 异步模型加载（不阻塞 GUI 线程）
3. 多模型支持（seed/base/large 三种规格，可动态切换）
4. 8 种基础情绪分类（愤怒、厌恶、恐惧、开心、平静、其他、悲伤、惊讶）
5. 多因子情绪稳定度评分（负面情绪权重 + Shannon 熵 + 极端度）
6. 复合情绪模式识别（焦虑、挫败、嫉妒、紧张、厌倦、愤恨）
7. 混合情绪检测（概率 > 8% 的并存情绪）
8. 个性化调节建议生成（分级建议策略）
9. 线程安全的状态管理

模型信息：
- 模型系列：emotion2vec+ (seed / base / large)
- 模型来源：ModelScope (阿里达摩院 DAMO Academy)
- 模型许可证：Apache License 2.0
- 支持分类：angry, disgusted, fearful, happy, neutral, other, sad, surprised
- 输入要求：16kHz 单声道 WAV 音频
- 推理方式：强制 CPU 推理，确保所有设备均可运行

算法原理：
- 稳定度评分：三因子加权模型（负面情绪权重 40% + 熵值 30% + 极端度 30%）
- 复合情绪：基于基础情绪组合模式 + 概率阈值触发
- 混合情绪：概率 > 8% 的非主要情绪自动检测

作者：Jiawei Li
许可证：GPL v3
"""

import os
import sys
import torch
import numpy as np
import logging
import threading
import warnings
import re

# 引入科研评估所需模块（P0/P1/P2）
from version import APP_VERSION, ALGORITHM_VERSION
from audio_quality import AudioQualityAnalyzer, compute_audio_quality
from audio_features import extract_acoustic_features, compute_psychological_indicators
from reliability import get_reliability_level
from baseline import PersonalBaseline

# 忽略一些不必要的警告信息，保持输出整洁
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------
# 环境变量预配置（必须在导入 modelscope/funasr 之前设置）
# ---------------------------------------------------------------------------
os.environ['FUNASR_AUTO_INSTALL'] = '0'
os.environ['FUNASR_INSTALL_DEP'] = '0'
os.environ['MODELSCOPE_AUTO_INSTALL_DEP'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger = logging.getLogger(__name__)


# ===========================================================================
# 常量定义
# ===========================================================================
# 以下常量定义了情绪识别系统的核心配置，包括：
# - 可用模型列表及元信息
# - 模型输出标签到统一中文标签的映射表
# - 情绪分类（正面/负面/中性）
# - 各情绪对稳定度的影响权重（基于心理学情绪维度理论）
# - 复合情绪模式定义（基于心理学情绪组合理论）
# - 稳定度等级定义与显示颜色
# - 稳定度计算中各因子的权重配置
# ===========================================================================

# 可用模型列表及其元信息（按规模从小到大排列）
AVAILABLE_MODELS = {
    "emotion2vec_plus_seed": {"display": "Seed（最小模型）", "size": "~200MB", "desc": "速度最快，适合低配置设备"},
    "emotion2vec_plus_base": {"display": "Base（基础模型）", "size": "~500MB", "desc": "速度与精度均衡"},
    "emotion2vec_plus_large": {"display": "Large（大型模型）", "size": "~1GB", "desc": "精度最高，推荐使用"},
}

# 模型原始标签到统一中文标签的映射表
# emotion2vec+ 模型可能返回中英文混合的标签（如 'angry'/'生气'/'愤怒' 都指同一情绪）
# 此映射表将所有可能的标签变体统一为 8 种标准中文标签
LABEL_MAPPING = {
    # 愤怒
    '生气': '愤怒',
    'angry': '愤怒',
    '愤怒': '愤怒',
    # 厌恶
    '厌恶': '厌恶',
    'disgusted': '厌恶',
    # 恐惧
    '恐惧': '恐惧',
    'fearful': '恐惧',
    '害怕': '恐惧',
    # 开心
    '开心': '开心',
    'happy': '开心',
    '高兴': '开心',
    '快乐': '开心',
    # 平静
    '中立': '平静',
    'neutral': '平静',
    '平静': '平静',
    # 其他
    '其他': '其他',
    'other': '其他',
    # 悲伤
    '难过': '悲伤',
    'sad': '悲伤',
    '悲伤': '悲伤',
    '伤心': '悲伤',
    # 惊讶
    '吃惊': '惊讶',
    'surprised': '惊讶',
    '惊讶': '惊讶',
    # 未知
    '<unk>': '其他'
}

# 情绪分类列表（基于心理学情绪维度理论）
# 负面情绪：对情绪稳定度有负面影响，权重用于稳定度计算
NEGATIVE_EMOTIONS = ['愤怒', '厌恶', '恐惧', '悲伤']
# 正面情绪：对稳定度无负面贡献，且可抑制极端度得分
POSITIVE_EMOTIONS = ['开心']
# 中性情绪：不直接影响稳定度核心得分
NEUTRAL_EMOTIONS = ['平静', '惊讶', '其他']

# 各情绪对情绪稳定度的影响权重（0-1）
# 权重设计依据：基于情绪的唤醒度（arousal）和效价（valence）维度
# 高唤醒 + 负效价 = 高不稳定权重（如恐惧 0.95、愤怒 0.90）
# 低唤醒 + 负效价 = 中等权重（如悲伤 0.80）
# 正效价/中性 = 零权重
EMOTION_WEIGHTS = {
    "愤怒": 0.90,    # 愤怒：高激活负面情绪
    "厌恶": 0.70,    # 厌恶：中等负面影响
    "恐惧": 0.95,    # 恐惧：高唤醒度，最不稳定
    "悲伤": 0.80,    # 悲伤：低激活但持续影响
    "惊讶": 0.30,    # 惊讶：短暂激活
    "开心": 0.0,     # 正面情绪无负面贡献
    "平静": 0.0,     # 中性无贡献
    "其他": 0.15     # 未识别轻微贡献
}

# 复合情绪模式定义
# 基于心理学情绪组合理论：复合情绪由多个基础情绪在特定模式下组合而成
# 当所有组成情绪的概率均超过阈值（COMPOUND_THRESHOLD=10%）时触发检测
# 多个复合情绪同时满足时，选择组成情绪总概率最高的作为最终结果
COMPOUND_EMOTIONS = {
    "焦虑": {
        "components": ["恐惧", "悲伤"],
        "desc": "恐惧与悲伤交织，表现为对未来的不确定感和持续担忧",
        "advice": "焦虑是恐惧和悲伤的复合体，建议通过正念呼吸和渐进式肌肉放松来缓解"
    },
    "挫败": {
        "components": ["愤怒", "悲伤"],
        "desc": "愤怒与悲伤并存，通常源于期望落空或目标受阻",
        "advice": "挫败感说明您对某事有期待，建议将大目标拆分为小步骤，逐步推进"
    },
    "嫉妒": {
        "components": ["愤怒", "悲伤", "恐惧"],
        "desc": "愤怒、悲伤和恐惧的三重交织，涉及对自身地位的不安",
        "advice": "这种复杂情绪需要自我接纳，建议关注自身成长而非与他人比较"
    },
    "紧张": {
        "components": ["恐惧", "惊讶"],
        "desc": "恐惧伴随警觉状态，面对未知挑战时的应激反应",
        "advice": "适度紧张有助于提升表现，建议通过准备和模拟来增加掌控感"
    },
    "厌倦": {
        "components": ["厌恶", "悲伤"],
        "desc": "厌恶与悲伤的结合，通常源于长期重复或缺乏意义感",
        "advice": "厌倦可能是需要变化的信号，试着为日常生活增添新的元素"
    },
    "愤恨": {
        "components": ["愤怒", "厌恶"],
        "desc": "愤怒叠加厌恶，强烈的排斥和对抗情绪",
        "advice": "这种强烈情绪需要安全释放，建议通过运动或书写来疏导"
    },
}

# 情绪稳定度等级定义（分数区间 → 等级名称 + 显示颜色）
STABILITY_LEVELS = {
    "非常稳定": "#27AE60",   # 绿色
    "良好": "#2ECC71",       # 浅绿
    "一般": "#F1C40F",       # 黄色
    "轻度波动": "#E67E22",   # 橙色
    "不稳定": "#E74C3C",     # 红色
    "情绪激烈": "#8B0000"    # 深红
}

# 稳定度计算中各因子的权重
# 综合公式：final = negative_weight * 0.40 + entropy * 0.30 + extremity * 0.30
# 三因子设计依据：
# - 负面情绪权重：直接反映负面情绪的强度
# - Shannon 熵：反映情绪分布的混乱程度，高熵=多情绪并存=心理冲突
# - 极端度：检测单一负面情绪的极端高概率状态
#
# ⚠️ 权重来源：专家设定（基于情绪维度理论）
# P2 数据驱动优化计划：
#   1. 收集 ≥30 名被试的语音情绪数据 + 标准化心理量表（PANAS/SAM/PHQ-9）
#   2. 以心理量表得分为因变量，三因子为自变量进行多元回归
#   3. 回归系数（标准化后）可作为新的因子权重
#   4. 使用交叉验证评估权重泛化性能
#   5. 最终权重需经过独立样本验证后更新
STABILITY_FACTOR_WEIGHTS = {
    "negative_weight": 0.40,   # 负面情绪加权分数权重
    "entropy": 0.30,           # 情绪分散度（Shannon 熵）权重
    "extremity": 0.30          # 情绪极端度权重
}


def normalize_label(raw_label):
    """
    将模型返回的原始标签标准化为统一的中文标签

    处理逻辑：
    1. 转换为小写字符串
    2. 去除可能的前缀（如 'emotion/angry' → 'angry'）
    3. 在 LABEL_MAPPING 中查找对应的中文标签
    4. 找不到则返回 '其他'

    参数：
        raw_label (str): 模型返回的原始标签

    返回值：
        str: 标准化后的中文标签（如 '愤怒'、'开心'、'平静' 等）
    """
    if not raw_label:
        return '其他'
    label_str = str(raw_label).strip().lower()
    # 处理可能的分隔符前缀（如 'emotion/angry' 或 'happy-sad'）
    for slash_pos in [label_str.find('/'), label_str.find('-'), label_str.find('_')]:
        if slash_pos > 0:
            label_str = label_str[:slash_pos]
            break
    label_str = label_str.strip()
    # 在映射表中查找，先尝试原格式，再尝试首字母大写
    return LABEL_MAPPING.get(label_str, LABEL_MAPPING.get(label_str.capitalize(), '其他'))


class EmotionRecognizer:
    """
    情绪识别器类（单例模式）

    全局唯一的情绪识别实例，负责：
    - 加载 emotion2vec+ 系列预训练模型（seed/base/large，可动态切换）
    - 对音频文件进行情绪识别推理
    - 计算情绪稳定度分数
    - 生成调节建议

    单例模式的原因：
    1. 模型加载耗时且占用内存大，全局只需加载一次
    2. 避免重复加载导致的资源浪费
    3. 统一管理模型加载状态

    线程安全性：
    - 使用双重检查锁定（Double-Checked Locking）保证单例线程安全
    - 模型加载过程使用独立锁，防止并发加载
    - 推理过程使用 torch.no_grad() 减少内存占用
    """

    _instance = None           # 单例实例
    _lock = threading.Lock()   # 单例创建锁

    def __new__(cls, *args, **kwargs):
        """
        单例模式的 __new__ 方法

        使用双重检查锁定确保线程安全：
        1. 第一次检查（无锁）：如果实例已存在，直接返回
        2. 获取锁
        3. 第二次检查（有锁）：如果实例仍不存在，创建新实例

        参数：
            *args, **kwargs: 传递给 __init__ 的参数

        返回值：
            EmotionRecognizer: 全局唯一实例
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, progress_callback=None, model_name=None):
        """
        初始化情绪识别器

        注意：由于是单例模式，__init__ 可能被调用多次，
        因此使用 _initialized 标志确保只初始化一次。

        参数：
            progress_callback (callable, 可选): 模型加载进度回调函数，
                接受一个字符串参数表示当前进度消息
            model_name (str, 可选): 要加载的模型名称，默认从配置文件读取
        """
        if self._initialized:
            # 已初始化，仅添加新的回调（如果有）
            if progress_callback:
                self._progress_callbacks.append(progress_callback)
            return
        self._initialized = True
        # 模型相关
        self.model = None           # 模型实例
        self.loaded = False         # 是否已加载完成
        self.loading = False        # 是否正在加载中
        self.error = None           # 加载错误信息
        # 模型名称（从配置文件读取或使用参数指定）
        from app_paths import load_model_config
        self.model_name = model_name or load_model_config()
        # 进度回调列表（支持多个回调）
        self._progress_callbacks = []
        if progress_callback:
            self._progress_callbacks.append(progress_callback)
        # 模型加载专用锁（防止并发加载）
        self._load_lock = threading.Lock()

    def _report_progress(self, msg):
        """
        报告加载进度（内部方法）

        遍历所有注册的进度回调并调用，
        单个回调失败不影响其他回调。

        参数：
            msg (str): 进度消息
        """
        logger.info(msg)
        for cb in list(self._progress_callbacks):
            try:
                if callable(cb):
                    cb(msg)
            except Exception:
                pass

    def add_progress_callback(self, callback):
        """
        添加进度回调函数

        参数：
            callback (callable): 回调函数，接受一个字符串参数
        """
        if callable(callback) and callback not in self._progress_callbacks:
            self._progress_callbacks.append(callback)

    def _fix_requirements_file(self, model_dir):
        """
        修复模型目录中的 requirements.txt 文件（内部方法）

        ModelScope 下载的模型可能包含带有特殊索引源的 requirements.txt，
        可能导致自动安装时出现问题。此函数清理这些内容，只保留纯包名。

        参数：
            model_dir (str): 模型目录路径
        """
        req_file = os.path.join(model_dir, 'requirements.txt')
        if not os.path.exists(req_file):
            return
        try:
            with open(req_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                line = line.strip()
                # 跳过空行和注释
                if not line or line.startswith('#'):
                    continue
                # 跳过索引源配置（如 -i https://...）
                if line.startswith('-i ') or line.startswith('--index-url '):
                    continue
                # 跳过其他以 - 开头的配置项
                if line.startswith('-'):
                    continue
                # 只保留以字母数字开头的包名
                if re.match(r'^[a-zA-Z0-9]', line):
                    new_lines.append(line)
            with open(req_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines) + '\n' if new_lines else '')
            logger.info("已修复模型依赖配置文件")
        except Exception as e:
            logger.warning(f"修复requirements.txt失败: {e}")
            # 兜底：清空文件内容
            try:
                with open(req_file, 'w', encoding='utf-8') as f:
                    f.write('')
            except Exception:
                pass

    @classmethod
    def reset_instance(cls):
        """
        重置单例实例（用于模型切换时重新创建实例）

        调用后下次实例化将创建新的 EmotionRecognizer 实例。
        """
        with cls._lock:
            cls._instance = None

    def load_model(self, callback=None):
        """
        加载情绪识别模型（异步）

        在独立线程中加载模型，不阻塞调用线程。
        加载完成后通过 callback 通知结果。

        加载流程：
        1. 检查是否已加载（已加载则直接返回成功）
        2. 检查是否正在加载（正在加载则忽略）
        3. 获取加载锁，再次检查（双重检查）
        4. 在新线程中执行实际加载
        5. 通过回调通知结果

        参数：
            callback (callable, 可选): 加载完成回调函数，签名为 callback(success, error)

        返回值：
            bool: True 表示已加载完成，False 表示正在加载中或加载失败
        """
        # 快速路径：已加载完成
        if self.loaded:
            if callback:
                try:
                    callback(True, None)
                except Exception:
                    pass
            return True
        # 快速路径：正在加载中
        if self.loading:
            if callback:
                self.add_progress_callback(lambda msg: None)
            return False

        # 获取加载锁，确保只有一个线程加载
        with self._load_lock:
            # 双重检查
            if self.loaded:
                if callback:
                    try:
                        callback(True, None)
                    except Exception:
                        pass
                return True
            if self.loading:
                return False
            self.loading = True

        def _do_load():
            """实际加载模型的线程函数"""
            try:
                from app_paths import get_model_cache_dir
                model_cache_dir = get_model_cache_dir()
                target_model_dir = os.path.join(model_cache_dir, 'models', 'iic', self.model_name)
                # 如果模型已存在，先修复其 requirements.txt
                if os.path.exists(target_model_dir):
                    self._fix_requirements_file(target_model_dir)

                self._report_progress("正在导入模型库...")
                from funasr import AutoModel

                model_display = AVAILABLE_MODELS.get(self.model_name, {}).get('display', self.model_name)
                self._report_progress(f"正在加载情绪识别模型 [{model_display}]（首次使用需要下载，之后启动会很快）...")
                # 使用 FunASR 的 AutoModel 加载模型
                self.model = AutoModel(
                    model=f"iic/{self.model_name}",
                    disable_pbar=True,       # 禁用进度条
                    disable_log=True,        # 禁用详细日志
                    # 优先使用 GPU，没有则使用 CPU
                    device="cpu"  # 强制使用CPU推理，确保所有电脑均可运行
                )

                self._report_progress("使用CPU进行推理...")

                self._report_progress(f"模型 [{model_display}] 加载完成！")
                self.loaded = True
                self.error = None
                if callback:
                    try:
                        callback(True, None)
                    except Exception:
                        pass
            except Exception as e:
                # 加载失败，清理状态
                self.model = None
                self.loaded = False
                self.error = str(e)
                self._report_progress(f"模型加载失败: {self.error}")
                logger.error(f"模型加载失败: {str(e)}", exc_info=True)
                if callback:
                    try:
                        callback(False, str(e))
                    except Exception:
                        pass
            finally:
                self.loading = False

        # 启动加载线程（守护线程，程序退出时自动结束）
        thread = threading.Thread(target=_do_load, daemon=True)
        thread.start()
        return False

    def switch_model(self, model_name, callback=None):
        """
        切换到指定模型

        如果目标模型与当前模型相同且已加载，直接返回成功。
        否则卸载当前模型，加载新模型。

        参数：
            model_name (str): 目标模型名称
            callback (callable, 可选): 加载完成回调，签名为 callback(success, error)

        返回值：
            bool: True 表示已立即完成（相同模型），False 表示正在异步加载
        """
        # 验证模型名称有效性
        if model_name not in AVAILABLE_MODELS:
            if callback:
                try:
                    callback(False, f"无效的模型名称: {model_name}")
                except Exception:
                    pass
            return False

        # 如果相同模型已加载，直接返回
        if model_name == self.model_name and self.loaded:
            if callback:
                try:
                    callback(True, None)
                except Exception:
                    pass
            return True

        # 如果正在加载，不允许切换
        if self.loading:
            if callback:
                try:
                    callback(False, "模型正在加载中，请稍后再试")
                except Exception:
                    pass
            return False

        # 清理当前模型
        logger.info(f"切换模型: {self.model_name} -> {model_name}")
        self.model = None
        self.loaded = False
        self.error = None
        self.model_name = model_name

        # 持久化配置
        from app_paths import save_model_config
        save_model_config(model_name)

        # 加载新模型
        return self.load_model(callback=callback)

    def _calculate_stability_score(self, probs_dict):
        """
        计算情绪稳定度分数（多因子综合评分）及分项

        计算逻辑综合三个因子：
        1. 负面情绪加权分数（权重 40%）：各情绪概率×影响权重
        2. 情绪分散度/熵（权重 30%）：概率分布的 Shannon 熵，高熵=不稳定
        3. 情绪极端度（权重 30%）：负面情绪是否极端高概率

        权重来源：专家设定（基于情绪维度理论），待 P2 阶段通过被试数据回归优化

        参数：
            probs_dict (dict): 各情绪的概率字典，键为情绪名称，值为概率（0-1）

        返回值：
            dict: 包含各因子分数和综合分数的字典
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
                "factor_weights_source": "专家设定（负面40% + 熵30% + 极端30%），基于情绪维度理论"
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
        # Shannon 熵计算，归一化到 0-10
        entropy = 0.0
        probs_list = []
        for prob in probs_dict.values():
            try:
                p = float(prob)
                if p > 0.001:  # 忽略极小概率
                    probs_list.append(p)
                    entropy -= p * math.log2(p)
            except (TypeError, ValueError):
                continue
        # 最大熵为 log2(N)，归一化
        max_entropy = math.log2(len(probs_dict)) if len(probs_dict) > 1 else 1.0
        entropy_normalized = (entropy / max_entropy) * 10.0 if max_entropy > 0 else 0.0
        entropy_score = round(min(10.0, max(0.0, entropy_normalized)), 2)

        # --- 因子 3: 情绪极端度 (0-10) ---
        # 检测是否存在极端高概率的负面情绪
        extremity_score = 0.0
        positive_total = 0.0
        for emotion, prob in probs_dict.items():
            try:
                prob_f = float(prob)
                if emotion in NEGATIVE_EMOTIONS:
                    if prob_f > 0.60:  # 单一负面情绪 > 60%
                        extremity_score = max(extremity_score, prob_f * 10)
                    elif prob_f > 0.40:  # 中等程度
                        extremity_score = max(extremity_score, prob_f * 7)
                if emotion in POSITIVE_EMOTIONS:
                    positive_total += prob_f
            except (TypeError, ValueError):
                continue
        # 正面情绪占主导时降低不稳定度
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

    def _get_stability_level(self, score):
        """
        根据稳定度分数获取情绪状态等级和颜色（内部方法）

        等级划分：
        - 0.0 - 2.0: 非常稳定（绿色）
        - 2.0 - 3.5: 良好（浅绿）
        - 3.5 - 5.0: 一般（黄色）
        - 5.0 - 6.5: 轻度波动（橙色）
        - 6.5 - 8.0: 不稳定（红色）
        - 8.0 - 10.0: 情绪激烈（深红）

        参数：
            score (float): 情绪稳定度分数（0-10）

        返回值：
            tuple: (等级名称, 颜色代码)
        """
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            return "未知", "#95A5A6"
        if score_f < 2.0:
            return "非常稳定", STABILITY_LEVELS["非常稳定"]
        elif score_f < 3.5:
            return "良好", STABILITY_LEVELS["良好"]
        elif score_f < 5.0:
            return "一般", STABILITY_LEVELS["一般"]
        elif score_f < 6.5:
            return "轻度波动", STABILITY_LEVELS["轻度波动"]
        elif score_f < 8.0:
            return "不稳定", STABILITY_LEVELS["不稳定"]
        else:
            return "情绪激烈", STABILITY_LEVELS["情绪激烈"]

    def _get_advice(self, main_emotion, stability_score, probs_dict):
        """
        根据情绪状态生成调节建议（内部方法）

        根据主要情绪类型和情绪稳定度分数，
        选择最合适的调节建议文本。

        建议分级策略：
        - 稳定度 < 2：非常好，给予正面反馈
        - 稳定度 2-3.5：轻微波动，温和建议
        - 稳定度 3.5-6.5：中度波动，较具体建议
        - 稳定度 >= 6.5：高度不稳定，紧急建议 + 寻求帮助提示

        参数：
            main_emotion (str): 主要情绪类型
            stability_score (float): 情绪稳定度分数
            probs_dict (dict): 所有情绪的概率分布

        返回值：
            str: 调节建议文本（带表情符号）
        """
        if stability_score < 2:
            if main_emotion == "开心":
                return "😊 您当前情绪状态非常好，继续保持这份好心情！"
            return "😊 您当前情绪状态稳定，心态平和，继续保持！"
        elif stability_score < 3.5:
            if main_emotion == "悲伤":
                return "💙 检测到些许低落情绪，可以试着做些让自己开心的事，听听音乐或和朋友聊聊天。"
            if main_emotion == "愤怒":
                return "❤️‍🔥 感觉到一些烦躁情绪，建议深呼吸放松一下，或者稍作休息。"
            if main_emotion == "惊讶":
                return "✨ 情绪有小幅波动，可能遇到了意想不到的事，深呼吸调整一下。"
            if main_emotion == "厌恶":
                return "🧡 感觉到轻微的不适，试着远离让您不舒服的环境，给自己一些空间。"
            return "🙂 情绪有轻微波动，属于正常范围，注意休息即可。"
        elif stability_score < 6.5:
            if main_emotion == "恐惧":
                return "💜 检测到紧张/焦虑情绪，建议找个安静的地方放松，必要时可以和信任的人倾诉。"
            if main_emotion == "悲伤":
                return "💙 情绪有些低落，不要独自承受，可以找人聊聊，或者做些放松的活动。"
            if main_emotion == "厌恶":
                return "🧡 感觉到一些抵触情绪，试着远离让您不适的事物，给自己一些空间。"
            if main_emotion == "愤怒":
                return "❤️‍🔥 情绪有些激动，建议先暂停当前事务，深呼吸10次，等平静后再处理问题。"
            if main_emotion == "惊讶":
                return "✨ 情绪波动较明显，可能受到了较大刺激，建议做些让自己平静的事情。"
            return "😐 情绪存在一定波动，建议适当休息，做一些放松活动。"
        else:
            if main_emotion == "恐惧":
                return "⚠️ 检测到较强的焦虑/恐惧情绪，建议立即停下休息，进行深呼吸放松，如果这种状态持续请寻求亲友陪伴或专业帮助。"
            if main_emotion == "悲伤":
                return "⚠️ 检测到较强的负面情绪，请不要独自承受，建议联系亲友倾诉，必要时寻求专业帮助。"
            if main_emotion == "愤怒":
                return "⚠️ 情绪较为激动，强烈建议先离开当前环境，进行深呼吸或冥想放松，避免在情绪激动时做决定。"
            if main_emotion == "厌恶":
                return "⚠️ 感觉到强烈的抵触情绪，建议立即更换环境，做些让自己舒适的事情，必要时和信任的人倒诉。"
            return "⚠️ 情绪波动较大，建议立即停下当前事务，进行深呼吸放松，必要时寻求他人陪伴。"

    def _analyze_compound_emotions(self, probs_dict, main_emotion):
        """
        分析复合情绪模式（内部方法）— P1 升级版：复合情绪强度评分

        使用强度评分公式（替代旧版固定阈值）：
            compound_score = min(components) × 0.5 + mean(components) × 0.3 + co_activation_bonus × 0.2
        其中 co_activation_bonus = 各组成情绪概率乘积的归一化值

        心理学依据：
        - 复合情绪由多种基础情绪组合而成（如焦虑=恐惧+悲伤）
        - co_activation_bonus 反映组成情绪同时激活的程度
        - "其他"情绪不参与复合情绪判断（非基础情绪类别）

        参数：
            probs_dict (dict): 各情绪的概率字典（包含所有8种情绪）
            main_emotion (str): 主要情绪类型

        返回值：
            dict 或 None: 复合情绪信息字典，包含：
                - name (str): 复合情绪名称
                - confidence (float): 复合情绪强度评分 (0-1)
                - components (dict): 各组成情绪及其概率
                - desc (str): 复合情绪描述
                - advice (str): 针对性建议
                - interpretation (str): 心理学解释
                未检测到时返回 None
        """
        if not isinstance(probs_dict, dict):
            return None

        # 不参与复合情绪判断的情绪类别
        EXCLUDED_EMOTIONS = {"其他"}

        best_match = None
        best_score = 0.0

        for compound_name, compound_info in COMPOUND_EMOTIONS.items():
            components = compound_info["components"]
            # 检查所有组成情绪是否都在排除列表外
            if any(c in EXCLUDED_EMOTIONS for c in components):
                continue

            # 获取各组成情绪的概率
            component_probs = []
            for comp_emotion in components:
                prob = float(probs_dict.get(comp_emotion, 0.0))
                component_probs.append(prob)

            # P1 升级：使用强度评分公式
            if len(component_probs) > 0:
                min_prob = min(component_probs)
                mean_prob = sum(component_probs) / len(component_probs)
                # co_activation_bonus: 所有组成情绪的概率乘积，反映"同时激活"程度
                co_activation = 1.0
                for p in component_probs:
                    co_activation *= max(p, 0.01)  # 避免乘以0
                co_activation_bonus = co_activation ** (1.0 / len(component_probs))  # 几何平均

                compound_score = min_prob * 0.5 + mean_prob * 0.3 + co_activation_bonus * 0.2

                if compound_score > best_score:
                    best_score = compound_score
                    best_match = compound_name

        # 只有当最佳得分超过最低阈值时才返回结果
        if best_match and best_score > 0.05:
            info = COMPOUND_EMOTIONS[best_match]
            component_details = {c: round(float(probs_dict.get(c, 0.0)), 4)
                               for c in info["components"]}
            return {
                "name": best_match,
                "confidence": round(best_score, 4),
                "components": component_details,
                "desc": info["desc"],
                "advice": info["advice"],
                "interpretation": f"由{'、'.join(info['components'])}组合而成，强度评分 {best_score:.2%}"
            }
        return None

    def predict_dual_model(self, audio_path):
        """
        科研模式：使用 Base + Large 双模型进行交叉验证

        使用当前模型作为主模型，同时使用另一个模型作为对照。
        比较两者结果的一致性，输出模型规模敏感性评估。

        参数：
            audio_path (str): 音频文件路径

        返回值：
            dict: 包含双模型结果的综合评估字典
        """
        if not self.loaded:
            return {"success": False, "error": "模型未加载"}

        # 确定对照模型
        if self.model_name == "emotion2vec_plus_large":
            secondary_model = "emotion2vec_plus_base"
        elif self.model_name == "emotion2vec_plus_base":
            secondary_model = "emotion2vec_plus_large"
        else:
            secondary_model = "emotion2vec_plus_base"  # seed 使用 base 作为对照

        # 主模型预测（标记为科研模式）
        primary_result = self.predict(audio_path, is_research_mode=True)
        if not primary_result.get('success'):
            return {"success": False, "error": f"主模型预测失败: {primary_result.get('error')}"}

        primary_result['model_name'] = self.model_name

        # 加载对照模型并预测
        try:
            original_model = self.model_name
            self.model = None
            self.loaded = False
            self.model_name = secondary_model
            self.load_model()
            # 等待加载完成（简化同步实现）
            import time
            waited = 0
            while not self.loaded and waited < 60:
                time.sleep(0.5)
                waited += 0.5

            if not self.loaded:
                return {"success": False, "error": f"对照模型 {secondary_model} 加载失败"}

            secondary_result = self.predict(audio_path)
            secondary_result['model_name'] = secondary_model

            # 恢复原模型
            self.model = None
            self.loaded = False
            self.model_name = original_model
            self.load_model()
            # 等待原模型加载完成，避免返回后下一次 predict() 因模型未加载而失败
            import time
            waited_restore = 0
            while not self.loaded and waited_restore < 60:
                time.sleep(0.5)
                waited_restore += 0.5

            # 评估模型一致性
            from reliability import evaluate_model_agreement
            agreement = evaluate_model_agreement(primary_result, secondary_result)

            # 以主模型结果为基础，叠加双模型验证信息，便于 GUI 直接展示
            dual_result = dict(primary_result)
            dual_result.update({
                "is_dual_model": True,
                "primary_model": original_model,
                "secondary_model": secondary_model,
                "model_agreement": agreement["agreement"],
                "result_reliability": agreement["level"],
                "model_agreement_note": agreement["note"],
                "secondary_result": secondary_result,
            })
            return dual_result

        except Exception as e:
            logger.error(f"双模型预测失败: {e}")
            # 至少返回主模型结果
            primary_result['is_dual_model'] = True
            primary_result['model_agreement'] = 0.0
            primary_result['result_reliability'] = "低"
            return primary_result

    def _generate_emotion_summary(self, main_emotion, confidence, mixed_emotions,
                                  compound_emotion, stability_score, stability_level):
        """
        生成情绪分析摘要文本（内部方法）

        综合主要情绪、复合情绪、混合情绪和稳定度，
        生成一段综合性的情绪状态描述。

        参数：
            main_emotion (str): 主要情绪
            confidence (float): 主要情绪置信度
            mixed_emotions (list): 混合情绪列表
            compound_emotion (dict or None): 复合情绪信息
            stability_score (float): 稳定度分数
            stability_level (str): 稳定度等级

        返回值：
            str: 情绪分析摘要文本
        """
        parts = []
        parts.append(f"主要情绪为「{main_emotion}」（置信度 {confidence:.0%}）")

        if compound_emotion:
            parts.append(f"检测到复合情绪「{compound_emotion['name']}」——{compound_emotion['desc']}")

        if mixed_emotions:
            mixed_str = "、".join([f"{e}({p*100:.0f}%)" for e, p in mixed_emotions[:3]])
            parts.append(f"伴随情绪：{mixed_str}")

        parts.append(f"情绪稳定度 {stability_score:.1f}/10（{stability_level}）")

        return "；".join(parts)

    def predict(self, audio_path, is_research_mode=False):
        """
        对音频文件进行情绪识别推理

        这是主要的对外接口，输入音频文件路径，
        返回包含情绪识别结果的字典。

        处理流程：
        1. 检查模型是否已加载
        2. 验证音频文件路径有效性
        3. 调用模型进行推理
        4. 解析模型输出（标签和分数）
        5. 标准化标签并计算概率分布
        6. 计算情绪稳定度分数和等级
        7. 生成调节建议
        8. 检测混合情绪
        9. 评估音频质量、提取声学特征、计算心理状态指标
        10. 评估综合可靠性并计算个人基线偏移（若已建立）

        参数：
            audio_path (str): 音频文件路径（WAV 格式，16kHz 单声道）
            is_research_mode (bool): 是否为科研评估模式，会影响元数据标记

        返回值：
            dict: 识别结果字典，包含以下字段：
                - success (bool): 是否成功
                - error (str): 错误信息（失败时）
                - 主要情绪 (str): 最主要的情绪类型
                - 置信度 (float): 主要情绪的置信度（0-1）
                - 所有情绪概率 (dict): 各情绪的概率分布
                - 完整概率_8类 (dict): 含"其他"的完整 8 类概率
                - 情绪稳定度分数 (float): 0-10分制稳定度评分
                - 情绪状态等级 (str): 稳定度等级名称
                - 等级颜色 (str): 等级对应的颜色代码
                - 稳定度分项 (dict): 三因子分项得分
                - valence_score (float): 效价估计值 [-1, 1]
                - arousal_score (float): 唤醒度估计值 [0, 1]
                - dominance_score (float): 掌控感估计值 [0, 1]
                - audio_quality (dict): 音频质量评估结果
                - acoustic_features (dict): 声学特征
                - psychological_indicators (dict): 心理状态指标
                - assessment_reliability (str): 综合可靠性等级
                - baseline_deviation (dict): 相对个人基线的偏移
                - 调节建议 (str): 个性化调节建议
                - 混合情绪 (list): 混合情绪列表 [(情绪, 概率), ...]
        """
        # 检查模型状态
        if not self.loaded:
            return {
                "success": False,
                "error": f"模型未加载: {self.error or '请等待模型加载完成'}"
            }

        # 验证输入路径
        if not audio_path or not isinstance(audio_path, str):
            return {
                "success": False,
                "error": "无效的音频文件路径"
            }

        if not os.path.exists(audio_path):
            return {
                "success": False,
                "error": f"音频文件不存在: {audio_path}"
            }

        if not os.path.isfile(audio_path):
            return {
                "success": False,
                "error": "指定路径不是有效的文件"
            }

        # 检查文件大小（小于 1KB 认为无效）
        file_size = 0
        try:
            file_size = os.path.getsize(audio_path)
        except OSError:
            pass
        if file_size < 1024:
            return {
                "success": False,
                "error": "音频文件过小，可能已损坏"
            }

        try:
            # 使用 no_grad 减少显存占用
            with torch.no_grad():
                result = self.model.generate(
                    audio_path,
                    granularity="utterance",   # 整句级别的情感识别
                    extract_embedding=False    # 不需要提取嵌入向量
                )

            # 处理返回结果（可能是列表，取第一个）
            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            if not result:
                return {
                    "success": False,
                    "error": "模型推理返回空结果"
                }

            # 提取标签和分数
            raw_labels = result.get("labels", [])
            raw_scores = result.get("scores", [])

            # 处理分数格式（可能是 numpy 数组或嵌套列表）
            if isinstance(raw_scores, np.ndarray):
                raw_scores = raw_scores.tolist()
            elif isinstance(raw_scores, list) and len(raw_scores) > 0 and isinstance(raw_scores[0], list):
                raw_scores = raw_scores[0]

            # 验证标签和分数数量是否匹配
            if not raw_labels or not raw_scores or len(raw_labels) != len(raw_scores):
                return {
                    "success": False,
                    "error": f"模型输出格式异常: labels={len(raw_labels)}, scores={len(raw_scores) if raw_scores else 0}"
                }

            # 初始化概率字典（所有情绪初始为0）
            probs_dict = {
                "愤怒": 0.0,
                "厌恶": 0.0,
                "恐惧": 0.0,
                "开心": 0.0,
                "平静": 0.0,
                "其他": 0.0,
                "悲伤": 0.0,
                "惊讶": 0.0
            }

            # 累加各情绪概率（同一中文标签可能对应多个原始标签）
            total_prob = 0.0
            for raw_label, raw_score in zip(raw_labels, raw_scores):
                try:
                    cn_label = normalize_label(raw_label)
                    prob = float(raw_score)
                    prob = max(0.0, min(1.0, prob))  # 限制在 0-1 范围
                    probs_dict[cn_label] = probs_dict.get(cn_label, 0.0) + prob
                    total_prob += prob
                except (TypeError, ValueError) as e:
                    logger.warning(f"解析分数失败 {raw_label}: {e}")
                    continue

            # 归一化概率（使总和为 1）
            if total_prob > 0.01:
                for k in probs_dict:
                    probs_dict[k] = probs_dict[k] / total_prob

            # 构建显示用的概率字典（排除"其他"类别，只展示主要情绪）
            display_probs = {
                "愤怒": probs_dict.get("愤怒", 0.0),
                "厌恶": probs_dict.get("厌恶", 0.0),
                "恐惧": probs_dict.get("恐惧", 0.0),
                "开心": probs_dict.get("开心", 0.0),
                "平静": probs_dict.get("平静", 0.0),
                "悲伤": probs_dict.get("悲伤", 0.0),
                "惊讶": probs_dict.get("惊讶", 0.0)
            }

            # 找出概率最高的情绪作为主要情绪
            max_label = max(display_probs.items(), key=lambda x: x[1])
            main_emotion = max_label[0]
            confidence = max_label[1]

            # 计算情绪稳定度（现在返回字典，含分项分数）
            stability_result = self._calculate_stability_score(probs_dict)
            stability_score = stability_result["stability_score"]
            stability_level = stability_result["stability_level"]
            level_color = stability_result["stability_color"]
            advice = self._get_advice(main_emotion, stability_score, probs_dict)

            # 计算 VAD 维度指标（从离散概率推导的估计值）
            from vad_dimensions import compute_vad_dimensions
            vad_dimensions = compute_vad_dimensions(probs_dict)

            # 检测混合情绪（概率 > 8% 且不是主要情绪的视为混合情绪）
            mixed_emotions = []
            for emo, prob in sorted(display_probs.items(), key=lambda x: -x[1]):
                if prob > 0.08 and emo != main_emotion:
                    mixed_emotions.append((emo, prob))

            # 复合情绪分析
            compound_emotion = self._analyze_compound_emotions(probs_dict, main_emotion)

            # 生成情绪分析摘要
            emotion_summary = self._generate_emotion_summary(
                main_emotion, confidence, mixed_emotions,
                compound_emotion, stability_score, stability_level
            )

            # =====================================================================
            # P0/P1/P2 升级：完整评估流程集成
            # =====================================================================
            # 1. 音频质量评估（保证输入一致性）
            audio_quality = None
            try:
                analyzer = AudioQualityAnalyzer(target_sr=16000)
                audio_quality = analyzer.analyze(audio_path)
            except Exception as e:
                logger.warning(f"音频质量评估失败: {e}")

            # 2. 声学特征提取（P1）
            acoustic_features = None
            try:
                acoustic_features = extract_acoustic_features(audio_path, sr=16000)
            except Exception as e:
                logger.warning(f"声学特征提取失败: {e}")

            # 3. 心理状态指标（P1）
            psychological_indicators = {}
            try:
                vad_dims = {
                    "valence_score": vad_dimensions["valence_score"],
                    "arousal_score": vad_dimensions["arousal_score"],
                    "dominance_score": vad_dimensions["dominance_score"],
                    "negative_load": vad_dimensions["negative_load"],
                    "emotional_uncertainty": vad_dimensions["emotional_uncertainty"],
                }
                psychological_indicators = compute_psychological_indicators(
                    acoustic_features, vad_dims
                )
            except Exception as e:
                logger.warning(f"心理状态指标计算失败: {e}")

            # 4. 综合可靠性评分（P0/P1）
            assessment_reliability = "中"
            try:
                reliability_assessment = {
                    "audio_quality": audio_quality,
                    "confidence": confidence,
                    "consistency": None,  # 单次检测无多次采样一致性
                }
                assessment_reliability = get_reliability_level(reliability_assessment)
            except Exception as e:
                logger.warning(f"可靠性评估失败: {e}")

            # 5. 个人基线偏移（P2）
            baseline_deviation = {"available": False, "note": "基线尚未建立"}
            try:
                baseline_mgr = PersonalBaseline()
                if baseline_mgr.is_established():
                    baseline_deviation = baseline_mgr.compute_deviation({
                        "主要情绪": main_emotion,
                        "置信度": confidence,
                        "情绪稳定度分数": stability_score,
                        "valence_score": vad_dimensions["valence_score"],
                        "arousal_score": vad_dimensions["arousal_score"],
                        "dominance_score": vad_dimensions["dominance_score"],
                        "negative_load": vad_dimensions["negative_load"],
                        "emotional_uncertainty": vad_dimensions["emotional_uncertainty"],
                        "acoustic_features": acoustic_features or {},
                    })
            except Exception as e:
                logger.warning(f"基线偏移计算失败: {e}")

            return {
                "success": True,
                "主要情绪": main_emotion,
                "置信度": round(confidence, 4),
                # 完整8类概率（含"其他"，用于科研记录）
                "完整概率_8类": {k: round(v, 4) for k, v in probs_dict.items()},
                # 显示用7类概率（排除"其他"，用于前端展示）
                "所有情绪概率": display_probs,
                # 原始模型输出（用于科研复算）
                "原始模型输出": {
                    "labels": [str(l) for l in raw_labels],
                    "scores": [round(float(s), 6) for s in raw_scores] if raw_scores else []
                },
                # 稳定度及分项（含权重来源标注）
                "情绪稳定度分数": stability_score,
                "情绪状态等级": stability_level,
                "等级颜色": level_color,
                "稳定度分项": {
                    "negative_weight_score": stability_result["negative_weight_score"],
                    "entropy_score": stability_result["entropy_score"],
                    "extremity_score": stability_result["extremity_score"],
                    "factor_weights_source": stability_result["factor_weights_source"]
                },
                # VAD 维度指标（⚠️ 从离散概率推导的估计值）
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
                "情绪分析摘要": emotion_summary,
                # P0/P1/P2 新增：音频质量、声学特征、心理状态指标、可靠性、基线
                "audio_quality": audio_quality if audio_quality else {},
                "acoustic_features": acoustic_features if acoustic_features else {},
                "psychological_indicators": psychological_indicators,
                "assessment_reliability": assessment_reliability,
                "baseline_deviation": baseline_deviation,
                # 科研元数据（可复现性）
                "model_name": self.model_name,
                "algorithm_version": ALGORITHM_VERSION,
                "app_version": APP_VERSION,
                "is_research_mode": bool(is_research_mode),
            }

        except torch.cuda.OutOfMemoryError:
            # GPU 显存不足
            logger.error("GPU显存不足")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return {
                "success": False,
                "error": "GPU显存不足，请关闭其他占用GPU的程序后重试"
            }
        except RuntimeError as e:
            # 运行时错误（可能是显存问题或其他）
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            logger.error(f"推理运行时错误: {error_msg}", exc_info=True)
            return {
                "success": False,
                "error": f"推理运行错误: {error_msg}"
            }
        except Exception as e:
            # 其他未知错误
            logger.error(f"推理过程出错: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"分析失败: {str(e)}"
            }

    def is_ready(self):
        """
        检查模型是否已就绪（可以进行推理）

        返回值：
            bool: True 表示模型已加载完成且可用
        """
        return self.loaded and self.model is not None

    def get_status(self):
        """
        获取当前模型状态

        返回值：
            str: 状态字符串，可能的值：
                - "ready": 已加载完成
                - "loading": 正在加载中
                - "error": 加载出错
                - "not_loaded": 未开始加载
        """
        if self.loaded:
            return "ready"
        elif self.loading:
            return "loading"
        elif self.error:
            return "error"
        else:
            return "not_loaded"
