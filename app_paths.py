# -*- coding: utf-8 -*-
"""
语音情绪识别系统 - 路径管理模块（便携模式）

本模块负责管理应用程序的所有数据存储路径，实现"便携模式"特性：
- 所有用户数据保存在程序目录的 portable_data 文件夹中
- 复制整个程序文件夹即可迁移所有数据和设置
- 若程序目录无写入权限，则自动回退到 %APPDATA%/VoiceMoodDetector
- 提供路径安全校验和安全删除功能，防止路径遍历攻击

主要功能：
1. 获取程序运行目录（兼容打包成 EXE 的情况）
2. 获取用户数据目录及各子目录（录音、模型、缓存、日志、临时文件等）
3. 配置 ModelScope 和 HuggingFace 的缓存路径
4. 路径安全校验（防止路径遍历攻击）
5. 安全文件删除（仅允许删除数据目录内的文件）
6. 存储空间统计

作者：Jiawei Li
许可证：MIT License
"""

import os
import sys
import json
import tempfile


def get_app_dir():
    """
    获取程序所在目录

    兼容两种运行模式：
    - 开发模式：返回脚本文件所在目录
    - 打包模式（PyInstaller）：返回 EXE 可执行文件所在目录

    返回值：
        str: 程序所在目录的绝对路径
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后的 EXE 运行模式
        return os.path.dirname(sys.executable)
    else:
        # 开发模式，直接运行 Python 脚本
        return os.path.dirname(os.path.abspath(__file__))


def get_user_data_dir():
    """
    获取用户数据根目录（便携模式）

    优先使用程序目录下的 portable_data 文件夹（便携模式）。
    如果该目录不可写（如安装在 Program Files），则回退到：
    - Windows: %APPDATA%/VoiceMoodDetector
    - 其他系统: 用户主目录/VoiceMoodDetector

    返回值：
        str: 用户数据目录的绝对路径（确保目录已存在）
    """
    path = os.path.join(get_app_dir(), 'portable_data')
    try:
        # 尝试创建目录并测试写入权限
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, '.write_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        # 写入测试成功，使用便携模式
        return path
    except (OSError, PermissionError, IOError):
        # 便携模式不可用，回退到系统用户目录
        appdata = os.environ.get('APPDATA', os.path.expanduser('~'))
        fallback_path = os.path.join(appdata, 'VoiceMoodDetector')
        os.makedirs(fallback_path, exist_ok=True)
        return fallback_path


def get_recordings_dir():
    """
    获取录音文件存储目录

    返回值：
        str: 录音文件目录的绝对路径（确保目录已存在）
    """
    path = os.path.join(get_user_data_dir(), 'recordings')
    os.makedirs(path, exist_ok=True)
    return path


def get_logs_dir():
    """
    获取日志文件存储目录

    返回值：
        str: 日志文件目录的绝对路径（确保目录已存在）
    """
    path = os.path.join(get_user_data_dir(), 'logs')
    os.makedirs(path, exist_ok=True)
    return path


def get_temp_dir():
    """
    获取临时文件存储目录

    用于存放程序运行过程中产生的临时文件，
    可通过"一键清理系统垃圾"功能删除。

    返回值：
        str: 临时文件目录的绝对路径（确保目录已存在）
    """
    app_temp = os.path.join(get_user_data_dir(), 'temp')
    os.makedirs(app_temp, exist_ok=True)
    return app_temp


def get_cache_dir():
    """
    获取缓存文件存储目录

    用于存放可加速程序启动的缓存文件，
    这些文件删除后会自动重建，不影响核心功能。

    返回值：
        str: 缓存文件目录的绝对路径（确保目录已存在）
    """
    path = os.path.join(get_user_data_dir(), 'cache')
    os.makedirs(path, exist_ok=True)
    return path


def get_history_path():
    """
    获取历史记录 JSON 文件路径

    返回值：
        str: history.json 文件的绝对路径
    """
    return os.path.join(get_user_data_dir(), 'history.json')


def get_model_cache_dir():
    """
    获取 AI 模型缓存目录

    用于存放 ModelScope 下载的预训练模型文件，
    如 emotion2vec_plus_large 等。

    返回值：
        str: 模型缓存目录的绝对路径（确保目录已存在）
    """
    path = os.path.join(get_user_data_dir(), 'models')
    os.makedirs(path, exist_ok=True)
    return path


def get_model_config_path():
    """
    获取模型配置文件路径

    返回值：
        str: model_config.json 文件的绝对路径
    """
    return os.path.join(get_user_data_dir(), 'model_config.json')


def load_model_config():
    """
    读取模型配置，返回当前选择的模型名

    返回值：
        str: 选中的模型名称，默认为 "emotion2vec_plus_large"
    """
    config_path = get_model_config_path()
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            model_name = config.get('selected_model', 'emotion2vec_plus_large')
            # 验证模型名合法性
            valid_models = ['emotion2vec_plus_seed', 'emotion2vec_plus_base', 'emotion2vec_plus_large']
            if model_name in valid_models:
                return model_name
    except (json.JSONDecodeError, IOError, OSError):
        pass
    return 'emotion2vec_plus_large'


def save_model_config(model_name):
    """
    保存模型配置

    参数：
        model_name (str): 模型名称

    返回值：
        bool: True 表示保存成功
    """
    config_path = get_model_config_path()
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({'selected_model': model_name}, f, ensure_ascii=False, indent=2)
        return True
    except (IOError, OSError):
        return False


def is_model_downloaded(model_name):
    """
    检查指定模型是否已下载

    参数：
        model_name (str): 模型名称（如 emotion2vec_plus_large）

    返回值：
        bool: True 表示模型已下载（model.pt 文件存在）
    """
    model_cache_dir = get_model_cache_dir()
    model_pt = os.path.join(model_cache_dir, 'models', 'iic', model_name, 'model.pt')
    return os.path.exists(model_pt)


def get_log_file():
    """
    获取主日志文件路径

    返回值：
        str: voice_mood_detect.log 文件的绝对路径
    """
    return os.path.join(get_logs_dir(), 'voice_mood_detect.log')


def get_preload_dir():
    """
    获取预加载资源目录

    用于存放程序启动时需要预加载的资源文件，
    该目录位于程序目录下，随程序一起分发。

    返回值：
        str: 预加载资源目录的绝对路径（确保目录已存在）
    """
    path = os.path.join(get_app_dir(), 'preload')
    os.makedirs(path, exist_ok=True)
    return path


def get_resource_path(relative_path):
    """
    获取资源文件的绝对路径（兼容打包模式）

    在 PyInstaller 打包后，资源文件会被解压到 _MEIPASS 临时目录，
    此函数会自动处理两种情况。

    参数：
        relative_path (str): 相对于程序根目录的资源路径

    返回值：
        str: 资源文件的绝对路径
    """
    if getattr(sys, 'frozen', False):
        # 打包模式：资源在 PyInstaller 的临时目录中
        base_path = sys._MEIPASS
    else:
        # 开发模式：资源在脚本所在目录
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def setup_modelscope_cache():
    """
    配置 ModelScope / HuggingFace / PyTorch 的缓存路径

    将所有 AI 模型缓存重定向到便携数据目录，
    确保模型文件与程序一起存储，便于迁移。

    注意：此函数必须在导入 modelscope/funasr 之前调用，
    否则环境变量设置不会生效。

    返回值：
        str: 模型缓存目录的绝对路径
    """
    model_dir = get_model_cache_dir()
    os.makedirs(model_dir, exist_ok=True)
    # ModelScope 缓存路径
    os.environ['MODELSCOPE_CACHE'] = model_dir
    os.environ['MODELSCOPE_HOME'] = model_dir
    # HuggingFace 缓存路径（部分模型可能通过 HF 下载）
    os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(model_dir, 'huggingface')
    # PyTorch 模型缓存
    os.environ['TORCH_HOME'] = os.path.join(model_dir, 'torch')
    return model_dir


def is_safe_path(filepath, base_dir=None):
    """
    校验路径安全性，防止路径遍历攻击

    检查给定的文件路径是否位于 base_dir 目录内部，
    防止通过 ../ 等方式越权访问系统其他文件。

    典型应用场景：
    - 删除用户上传/指定的文件前校验
    - 读取用户数据文件前校验

    参数：
        filepath (str): 待校验的文件路径
        base_dir (str, 可选): 基准目录，默认为用户数据目录

    返回值：
        bool: True 表示路径安全（在基准目录内），False 表示不安全
    """
    if base_dir is None:
        base_dir = get_user_data_dir()
    try:
        abs_file = os.path.abspath(filepath)
        abs_base = os.path.abspath(base_dir)
        # 公共前缀必须等于基准目录，说明文件在目录内部
        return os.path.commonpath([abs_file, abs_base]) == abs_base
    except (ValueError, OSError):
        return False


def safe_remove_file(filepath):
    """
    安全删除文件

    仅允许删除用户数据目录内的文件，防止误删系统文件。
    删除前会进行路径安全校验。

    参数：
        filepath (str): 要删除的文件路径

    返回值：
        bool: True 表示删除成功（或文件不存在），False 表示删除失败或路径不安全
    """
    if not filepath or not isinstance(filepath, str):
        return False
    try:
        # 文件不存在视为成功（幂等性）
        if not os.path.exists(filepath):
            return True
        # 路径安全校验
        if not is_safe_path(filepath, get_user_data_dir()):
            return False
        # 仅删除文件，不删除目录
        if os.path.isfile(filepath):
            os.remove(filepath)
            return True
    except (OSError, PermissionError, IOError):
        pass
    return False


def get_storage_stats():
    """
    获取存储空间统计信息

    统计各数据子目录的文件数量和占用空间大小，
    用于数据管理界面展示。

    返回值：
        dict: 包含各目录统计信息的字典，键为目录名，值为包含以下字段的字典：
            - path (str): 目录绝对路径
            - size_bytes (int): 总大小（字节）
            - size_mb (float): 总大小（MB）
            - file_count (int): 文件数量
    """
    stats = {}
    data_dir = get_user_data_dir()
    # 统计各个子目录
    for subdir in ['recordings', 'models', 'logs', 'temp', 'cache']:
        dir_path = os.path.join(data_dir, subdir)
        total_size = 0
        file_count = 0
        if os.path.exists(dir_path):
            # 递归遍历统计所有文件
            for root, dirs, files in os.walk(dir_path):
                for f in files:
                    try:
                        fp = os.path.join(root, f)
                        total_size += os.path.getsize(fp)
                        file_count += 1
                    except (OSError, PermissionError):
                        pass
        stats[subdir] = {
            'path': dir_path,
            'size_bytes': total_size,
            'size_mb': total_size / (1024 * 1024),
            'file_count': file_count
        }
    # 单独统计历史记录文件
    history_path = get_history_path()
    history_size = 0
    if os.path.exists(history_path):
        try:
            history_size = os.path.getsize(history_path)
        except OSError:
            pass
    stats['history'] = {
        'path': history_path,
        'size_bytes': history_size,
        'size_mb': history_size / (1024 * 1024)
    }
    return stats
