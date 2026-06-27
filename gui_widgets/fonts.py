# -*- coding: utf-8 -*-
"""
字体集中定义模块

统一管理界面字体族，避免字体名散落在各处。Qt 的 QFont 接受逗号分隔的
字体族列表，会按顺序自动回退到系统中第一个可用的字体。

字体选择理由：
- UI_FONT：中文「微软雅黑 UI」（字形比 Microsoft YaHei 更舒展、更现代，
  Windows 自带）→ 回退「微软雅黑」→ 「黑体」。
- MONO_FONT：等宽字体，用于数字展示。「Cascadia Code」（微软新一代等宽
  字体，比 Consolas 更清晰易读，Win10+ 自带）→ 回退「Consolas」→
  「Courier New」。

作者：Jiawei Li
许可证：GPL v3
"""

# 中英文 UI 字体族（带回退）
UI_FONT = "Microsoft YaHei UI, Microsoft YaHei, SimHei"

# 等宽数字字体族（带回退）
MONO_FONT = "Cascadia Code, Consolas, Courier New, monospace"
