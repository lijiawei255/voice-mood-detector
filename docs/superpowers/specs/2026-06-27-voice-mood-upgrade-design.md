# Voice Mood Detect — 科研级语音心理状态评估平台升级设计

> 日期：2026-06-27 | 版本：v1.0 | 状态：待实现

## 一、概述

基于现有语音情绪识别系统，升级为标准化、可解释、可复现的语音心理状态评估平台。为后续个体化声音刺激治疗科研项目提供可靠的检测基础。

### 技术环境
- Python: Anaconda `audio` 环境
- 推理: CPU only（确保跨设备一致性）
- GUI: PyQt5 + matplotlib
- 模型: emotion2vec+ (seed/base/large)

## 二、新增模块

### P0 新增模块

#### `audio_quality.py` — 音频质量评估
- VAD（语音活动检测）基于 librosa
- 自动裁剪前后静音
- 音量归一化
- 简单降噪（谱减法）
- 音频质量指标：duration, speech_duration, speech_ratio, rms_mean, rms_std, clipping_ratio, noise_level, quality_score

#### `vad_dimensions.py` — 效价-唤醒度-掌控感维度估计
- 从离散情绪概率映射到连续 VAD 维度
- Valence (-1.0 ~ 1.0), Arousal (0.0 ~ 1.0), Dominance (0.0 ~ 1.0)
- 负性负荷 (Negative Load)
- 情绪不确定性 (基于 Shannon 熵)
- ⚠️ 标注为"从离散概率推导的估计值"，非独立科研变量

### P1 新增模块

#### `audio_features.py` — 声学特征提取
- 使用 librosa + praat-parselmouth
- 特征: F0 (平均/波动), 音量 (平均/方差), 语速, 静音比例, 短时能量, MFCC, 频谱质心, jitter, shimmer, HNR
- 声音-心理状态映射: stress_index, anxiety_index, depression_tendency_index, emotional_activation, speech_stability

#### `reliability.py` — 可靠性评估
- 测试-重测信度 ICC 计算
- 双模型一致性检验（重定义为"模型规模敏感性检验"）
- 多次采样综合评估（均值/标准差/一致性）
- 综合可靠性等级

#### `export_manager.py` — 数据导出
- CSV 导出（SPSS/Excel 兼容）
- JSON 研究数据集导出
- 完整元数据记录

### P2 新增模块

#### `baseline.py` — 个人基线建模
- 首次使用采集 3-5 条平静语音建立基线
- 基线指标：平均音高、音量范围、语速、平静概率分布、平均稳定度
- 输出：baseline_deviation, current_state_vs_baseline, personalized_stability_score

#### `statistics.py` — 统计分析
- 效应量计算 (Cohen's d)
- 统计检验 (t-test, ANOVA if applicable)
- 情绪分布统计
- 异常波动检测

## 三、现有模块修改

### emotion_recognizer.py
| 改动 | 说明 | 优先级 |
|------|------|--------|
| 稳定度分项输出 | 返回 negative_weight_score, entropy_score, extremity_score, stability_level | P0 |
| 复合情绪强度评分 | compound_score = min(components)*0.5 + mean*0.3 + co_activation*0.2 | P0 |
| VAD 维度输出 | valence, arousal, dominance, negative_load, emotional_uncertainty | P0 |
| 原始模型输出保存 | raw_model_output（labels/scores 全量） | P0 |
| 双模型推理模式 | 科研模式使用 Base + Large 双模型评估 | P1 |

### gui.py + 新增 gui_widgets/
| 改动 | 说明 | 优先级 |
|------|------|--------|
| 录音质量实时反馈 | 音量指示、噪声警告、有效语音比例 | P0 |
| 结构化评估报告卡片 | 基础情绪、维度指标、心理状态、质量、可靠性 | P0 |
| 科研评估模式切换 | 快速检测/科研评估双模式 | P1 |
| 维度雷达图 | Valence-Arousal-Dominance 可视化 | P1 |
| 历史统计面板 | 情绪分布、复合情绪频次、稳定度趋势、周/月视图 | P1 |
| 研究数据导出按钮 | CSV/JSON 导出入口 | P1 |
| 个人基线管理界面 | 基线采集、查看、重置 | P2 |
| 统计分析面板 | 效应量、趋势检验结果展示 | P2 |

### history_manager.py
| 改动 | 说明 | 优先级 |
|------|------|--------|
| 扩展记录 schema | raw_model_output, probs_8, probs_7, vad_scores, audio_quality, stability_factors, model_name, algorithm_version, app_version, is_research_mode | P0 |
| 科研模式禁用清理 | 科研模式下不触发 AUTO_CLEAN | P0 |
| 统计 API | get_statistics(), get_emotion_distribution(), get_stability_summary() | P1 |
| 导出 API | export_records_csv(), export_research_dataset() | P1 |

### recorder.py
| 改动 | 说明 | 优先级 |
|------|------|--------|
| 实时音量统计 | RMS 均值/方差，clipping 检测 | P0 |
| 噪声水平估计 | 录音前静音段噪声估计 | P1 |

### main.py
| 改动 | 说明 | 优先级 |
|------|------|--------|
| 版本常量 | ALGORITHM_VERSION, APP_VERSION | P0 |

## 四、数据流

```
Audio Input
  ├─→ [录音质量实时监控] → 音量/噪声/有效语音反馈
  └─→ [保存音频文件]
       └─→ [音频质量评估] → audio_quality dict
            └─→ [声学特征提取] → acoustic_features dict (P1)
                 └─→ [情绪模型推理] → raw_model_output
                      ├─→ [VAD 维度估计] → valence/arousal/dominance
                      ├─→ [稳定度计算] → stability + sub-scores
                      ├─→ [复合情绪评分] → compound + intensity
                      └─→ [基线对比] → deviation (P2)
                           └─→ [可靠性评估] → reliability (P1)
                                └─→ 结构化评估报告
                                     └─→ 历史记录 (完整元数据)
```

## 五、GUI 布局变更

### 检测页面（快速模式）
```
┌──────────────────────────────────────────────────────────┐
│  [快速检测] [科研评估]             模型选择 [Large ▼]    │
├──────────────────────────┬───────────────────────────────┤
│   录音区域                │   快速指南                    │
│   [🎤 开始录音]           │                               │
│   时长: 12.3s             │                               │
│   录音质量: 良好 ████░░   │                               │
│   音量: ████░░░░          │                               │
│                          │                               │
│   检测结果                │                               │
│   ┌──────────────────┐   │                               │
│   │ 主要情绪: 平静    │   │                               │
│   │ 稳定度: 2.1/10   │   │                               │
│   │ 效价: +0.3       │   │                               │
│   │ 唤醒度: 0.2      │   │                               │
│   └──────────────────┘   │                               │
│                          │                               │
│   ┌─ 维度雷达图 ─────┐   │                               │
│   │    (matplotlib)    │   │                               │
│   └──────────────────┘   │                               │
├──────────────────────────┴───────────────────────────────┤
│  [概率柱状图]                                               │
└──────────────────────────────────────────────────────────┘
```

### 检测页面（科研模式）
额外显示：
- 环境噪声检测 3 秒倒计时
- 统一提示语显示
- 质量门控状态（通过/未通过）
- 多段录音进度（1/3, 2/3, 3/3）
- 综合评估报告

## 六、实现顺序

### Phase 1: P0 核心基础设施 (session 1-2)
1. `audio_quality.py` 模块
2. `vad_dimensions.py` 模块
3. `emotion_recognizer.py` — 稳定度分项 + VAD 输出 + 原始输出保存
4. `history_manager.py` — 扩展 schema + 科研模式禁用清理
5. `main.py` — 版本常量
6. `gui.py` — 录音质量反馈 + 结果卡片升级
7. `gui_widgets/result_cards.py` — 结构化评估卡片

### Phase 2: P1 科研增强 (session 3-4)
1. `audio_features.py` 模块
2. `reliability.py` 模块
3. `export_manager.py` 模块
4. `emotion_recognizer.py` — 复合情绪强度 + 双模型模式
5. `gui.py` — 科研模式 UI + 雷达图 + 统计面板
6. `history_manager.py` — 统计 API + 导出

### Phase 3: P2 高级功能 (session 5-6)
1. `baseline.py` 模块
2. `statistics.py` 模块
3. `gui.py` — 基线管理 + 统计分析面板
4. `emotion_recognizer.py` — 权重标注来源

## 七、测试策略

- 每个新模块独立单元测试
- `emotion_recognizer.py` 回归测试
- GUI 手动截图验证（使用视觉 MCP）
- 历史数据向后兼容测试
- 端到端录音→分析→保存流程测试
