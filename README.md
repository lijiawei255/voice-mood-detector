# 🎙️ 语音情绪识别系统 (Voice Mood Detector)

[![GPL v3 License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![ModelScope](https://img.shields.io/badge/AI-ModelScope-purple.svg)](https://www.modelscope.cn/)

[English](README.en.md) | 中文

---

## 📖 项目简介

语音情绪识别系统是一款基于深度学习的桌面应用程序，通过麦克风录制用户的语音，使用 AI 模型智能识别语音中的情绪状态，评估情绪稳定度，并给出个性化的调节建议。

系统采用**便携模式**设计，所有用户数据（录音、模型、历史记录、日志等）都保存在程序文件夹中，复制整个程序文件夹即可迁移所有数据，不污染系统目录。

> ⚠️ **免责声明**：本软件仅供个人非商用参考使用，检测结果仅作为情绪状态的辅助参考，不构成任何医疗诊断或建议。如有持续的情绪困扰，请及时咨询专业心理医生。

---

## ✨ 功能特性

| 功能 | 说明 |
|------|------|
| 🎤 实时录音 | 支持实时麦克风录音，显示录音时长和音量 |
| 🧠 智能识别 | 基于 emotion2vec_plus_large 模型，识别 7 种情绪 |
| 📊 情绪稳定度 | 0-10分制评分，评估情绪波动程度 |
| 🎯 混合情绪检测 | 自动识别混合情绪，更贴近真实状态 |
| 💡 调节建议 | 根据情绪状态给出个性化的调节建议 |
| 📋 历史记录 | 自动保存历史记录，最多保留 200 条 |
| 📈 趋势图表 | 可视化情绪稳定度变化趋势 |
| 💾 便携模式 | 所有数据保存在程序文件夹，即插即用 |
| 🧹 一键清理 | 一键清理临时文件、缓存、日志等系统垃圾 |

---

## 🚀 快速开始

### 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10 / Windows 11 |
| Python 版本 | 3.8 ~ 3.11 |
| 内存 | 建议 4GB 以上 |
| 磁盘空间 | 至少 2GB（模型约 1GB + 运行空间） |
| 音频设备 | 可用的麦克风 |

### 安装步骤

1. **克隆或下载项目**

```bash
git clone <仓库地址>
cd Voice_Mood_Detect
```

2. **创建虚拟环境（推荐）**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

> 💡 **提示**：如果 PyAudio 安装失败，可以尝试使用 conda 安装：
> ```bash
> conda install pyaudio -y
> ```

### 运行程序

```bash
python main.py
```

---

## 📖 使用指南

### 首次运行（模型下载）

1. 首次启动程序时，会显示**欢迎向导**，介绍系统功能和使用方法
2. 点击「🚀 开始使用（下载模型）」按钮
3. 程序会自动从 ModelScope 下载 **emotion2vec_plus_large** 模型
4. 模型大小约 **1GB**，请耐心等待下载完成
5. 下载完成后模型会保存在 `portable_data/models/` 目录中
6. 以后启动程序会直接加载本地模型，无需重新下载

### 录音与分析

1. 等待模型加载完成（状态栏显示「● 准备就绪」）
2. 点击绿色的 **「🎤 点击开始录音」** 按钮
3. 自然地说出您现在的感受（建议录音时长 3-30 秒）
4. 点击红色的 **「⏹️ 点击停止录音」** 按钮
5. 稍等几秒，即可看到分析结果

### 结果说明

- **情绪稳定度**：0-10 分，分数越高表示情绪波动越大
  - 0-2 分：非常稳定 🟢
  - 2-4 分：良好 🟢
  - 4-6 分：一般 🟡
  - 6-8 分：不稳定 🔴
  - 8-10 分：情绪激烈 🔴

- **主要情绪**：平静、开心、惊讶、悲伤、愤怒、恐惧、厌恶
- **混合情绪**：同时检测到的其他情绪（概率 > 8%）
- **调节建议**：根据情绪状态给出的个性化建议

### 历史记录

1. 点击顶部的 **「📋 历史报告」** 标签页
2. 查看所有历史检测记录（按时间倒序排列）
3. 下方显示**情绪稳定度变化趋势图**
4. 记录序号按时间先后排列（第 1 次、第 2 次...）

### 数据管理

1. 点击顶部菜单 **「文件 → 数据管理」**（快捷键 Ctrl+D）
2. 查看数据存储位置和存储空间统计
3. 支持的操作：
   - **清空所有历史记录和录音**：删除所有检测记录和音频文件
   - **一键清理系统垃圾**：清除临时文件、缓存文件、日志文件
   - **打开数据文件夹**：在文件管理器中打开数据目录

### 程序卸载

由于采用便携模式，卸载非常简单：

1. 关闭程序
2. 直接删除整个程序文件夹即可
3. 不会在系统中留下任何残留文件

> 💡 如果只需要清除个人数据而保留程序，可以使用「数据管理」中的「清空历史记录」和「一键清理系统垃圾」功能。

---

## 🛠️ 技术栈

### 核心技术

| 技术 | 用途 | 版本/说明 |
|------|------|-----------|
| **Python** | 开发语言 | 3.8+ |
| **PyQt5** | GUI 框架 | GPL v3 许可 |
| **PyTorch** | 深度学习框架 | 模型推理 |
| **ModelScope** | 模型库管理 | 阿里达摩院 |
| **FunASR** | 语音识别框架 | 支持 emotion2vec |
| **emotion2vec_plus_large** | 情绪识别模型 | Apache 2.0 许可 |
| **PyAudio** | 音频录制 | MIT 许可 |
| **librosa** | 音频处理 | ISC 许可 |
| **matplotlib** | 数据可视化 | 绘制趋势图 |
| **NumPy** | 数值计算 | 数据处理 |

### 项目架构

```
Voice_Mood_Detect/
├── main.py                 # 程序入口
├── gui.py                  # 图形界面（主窗口、对话框、控件）
├── recorder.py             # 音频录制模块
├── emotion_recognizer.py   # 情绪识别核心模块
├── history_manager.py      # 历史记录管理
├── app_paths.py            # 路径管理（便携模式）
├── relaxation_tips.py      # 调节建议库
├── requirements.txt        # Python 依赖列表
├── LICENSE                 # GPL v3 开源协议
├── README.md               # 中文说明文档
├── README.en.md            # 英文说明文档
└── .gitignore              # Git 忽略配置
```

### 运行时目录

程序运行后会自动创建以下目录（便携模式）：

```
portable_data/
├── recordings/    # 录音文件（可删除）
├── models/        # AI 模型文件（不可删除，约1GB）
├── logs/          # 运行日志（可清除）
├── temp/          # 临时文件（可清除）
├── cache/         # 缓存文件（可清除）
└── history.json   # 历史记录（可删除）
```

---

## 📄 开源协议

### 本项目协议

本项目采用 **GNU General Public License v3.0 (GPL v3)** 开源协议。

选择 GPL v3 的原因：
- 本项目使用 PyQt5（GPL v3 许可），为保持许可证一致性采用 GPL v3
- 确保项目及其衍生作品始终保持开源
- 保护开源社区的共同利益

完整协议文本请参阅 [LICENSE](LICENSE) 文件。

### 第三方依赖及协议

| 项目 | 许可证 | 说明 |
|------|--------|------|
| **emotion2vec_plus_large** | Apache License 2.0 | 情绪识别模型，由 ModelScope 提供 |
| **FunASR** | Apache License 2.0 | 语音识别工具包 |
| **ModelScope** | Apache License 2.0 | 模型服务平台 |
| **PyTorch** | BSD-style | 深度学习框架 |
| **PyQt5** | GPL v3 / 商业 | GUI 框架（开源版使用 GPL v3） |
| **PyAudio** | MIT | 音频录制库 |
| **librosa** | ISC | 音频处理库 |
| **matplotlib** | PSF-based | 数据可视化库 |
| **NumPy** | BSD | 数值计算库 |

> ✅ **兼容性说明**：以上所有第三方库的许可证均与 GPL v3 兼容。

---

## 🤝 贡献指南

欢迎对本项目做出贡献！您可以通过以下方式参与：

1. **提交 Bug 报告**：在 Issues 中描述遇到的问题
2. **提出功能建议**：分享您的想法和建议
3. **提交代码**：Fork 项目后提交 Pull Request
4. **完善文档**：帮助改进文档和翻译

### 开发环境搭建

```bash
# 克隆项目
git clone <仓库地址>
cd Voice_Mood_Detect

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 安装开发依赖
pip install -r requirements.txt

# 运行程序
python main.py
```

### 代码规范

- 所有 Python 代码使用 UTF-8 编码
- 核心模块需包含详细的中文注释和 docstring
- 遵循 PEP 8 代码风格
- 提交前请确保代码可以正常运行

---

## ⚠️ 免责声明

1. 本软件仅供个人学习和非商用参考使用
2. 情绪检测结果仅作参考，不构成任何医疗诊断或建议
3. 如有持续的情绪低落、焦虑或其他心理困扰，请及时咨询专业心理医生
4. 使用本软件产生的任何后果由使用者自行承担
5. AI 模型由第三方提供，本项目不对模型的准确性和可靠性作任何保证

---

## 📞 联系方式

- 项目作者：**Jiawei Li**
- 项目仓库：GitHub

---

*感谢使用语音情绪识别系统！如果觉得有用，请给个 Star ⭐ 支持一下~*
