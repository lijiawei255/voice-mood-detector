# -*- coding: utf-8 -*-
"""
语音情绪识别系统 - 程序入口模块

本模块是整个应用程序的启动入口，负责完成以下工作：
1. 设置环境变量，禁用第三方库的自动安装和并行警告
2. 初始化模型缓存路径（便携模式，所有数据存在程序目录）
3. 配置 PyQt5 应用的高 DPI 支持和全局字体
4. 创建并显示主窗口，启动 Qt 事件循环

启动流程（顺序关键）：
    1. 环境变量预配置（必须在 import modelscope/funasr 之前）
    2. ModelScope/HuggingFace/PyTorch 缓存路径重定向
    3. PyQt5 高 DPI 属性设置（必须在 QApplication 创建前）
    4. QApplication 实例创建 + 全局字体配置
    5. 主窗口创建与显示
    6. Qt 事件循环启动

运行方式：
    python main.py

作者：Jiawei Li
许可证：GPL v3
"""

import sys
import os

# ===========================================================================
# 环境变量预配置（必须在导入相关库之前设置）
# ===========================================================================
# 这些环境变量控制 FunASR/ModelScope 的行为，防止运行时自动下载依赖
# 必须在 `from funasr import ...` 或 `from modelscope import ...` 之前设置
# ===========================================================================
# 禁用 FunASR 的自动安装功能，避免运行时尝试下载依赖
os.environ['FUNASR_AUTO_INSTALL'] = '0'
os.environ['FUNASR_INSTALL_DEP'] = '0'
# 禁用 ModelScope 的自动依赖安装
os.environ['MODELSCOPE_AUTO_INSTALL_DEP'] = '0'
# 禁用 tokenizers 的并行警告，减少不必要的控制台输出
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 导入路径管理模块并设置模型缓存目录（便携模式）
# setup_modelscope_cache() 将 ModelScope/HuggingFace/PyTorch 缓存重定向到 portable_data/models/
# 必须在导入 modelscope/funasr 之前完成，否则环境变量不会生效
from app_paths import setup_modelscope_cache, get_log_file
from version import APP_VERSION, ALGORITHM_VERSION
setup_modelscope_cache()

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from gui import MainWindow


def main():
    """
    程序主入口函数

    完成以下初始化工作：
    1. 启用高 DPI 缩放和高分辨率图标，提升高分屏显示效果
    2. 创建 QApplication 实例，设置应用名称和组织名
    3. 设置全局字体为微软雅黑，确保中文显示正常
    4. 创建主窗口并以最大化方式显示
    5. 启动 Qt 事件循环

    返回值：
        程序退出码（0 表示正常退出）
    """
    # 启用高 DPI 支持（必须在创建 QApplication 之前设置）
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # 创建应用实例
    app = QApplication(sys.argv)
    app.setApplicationName("语音情绪识别系统")
    app.setOrganizationName("VoiceMoodDetector")
    # 设置全局字体，优先使用微软雅黑以保证中文显示效果
    app.setFont(QFont("Microsoft YaHei", 10))

    # 创建并显示主窗口（默认最大化）
    window = MainWindow()
    window.showMaximized()

    # 启动事件循环，直到用户关闭程序
    sys.exit(app.exec_())


if __name__ == "__main__":
    # 当直接运行本脚本时启动程序
    main()
