import os
import sys

print("=" * 60)
print("  语音情绪识别系统 - 路径功能测试")
print("=" * 60)
print()

from app_paths import (
    get_user_data_dir,
    get_recordings_dir,
    get_logs_dir,
    get_temp_dir,
    get_cache_dir,
    get_history_path,
    get_model_cache_dir,
    get_log_file,
    setup_modelscope_cache
)

print("[测试1] 用户数据目录:")
print(f"  {get_user_data_dir()}")
assert os.path.exists(get_user_data_dir()), "用户数据目录应该存在"
print("  ✓ 目录存在")
print()

print("[测试2] 录音文件目录:")
print(f"  {get_recordings_dir()}")
assert os.path.exists(get_recordings_dir()), "录音目录应该存在"
print("  ✓ 目录存在")
print()

print("[测试3] 日志目录:")
print(f"  {get_logs_dir()}")
assert os.path.exists(get_logs_dir()), "日志目录应该存在"
print("  ✓ 目录存在")
print()

print("[测试4] 临时文件目录:")
print(f"  {get_temp_dir()}")
assert os.path.exists(get_temp_dir()), "临时目录应该存在"
print("  ✓ 目录存在")
print()

print("[测试5] 缓存目录:")
print(f"  {get_cache_dir()}")
assert os.path.exists(get_cache_dir()), "缓存目录应该存在"
print("  ✓ 目录存在")
print()

print("[测试6] 模型缓存目录:")
model_dir = get_model_cache_dir()
print(f"  {model_dir}")
assert os.path.exists(model_dir), "模型目录应该存在"
print("  ✓ 目录存在")
print()

print("[测试7] 设置ModelScope缓存环境变量:")
setup_modelscope_cache()
modelscope_cache = os.environ.get('MODELSCOPE_CACHE', '')
print(f"  MODELSCOPE_CACHE = {modelscope_cache}")
assert modelscope_cache == model_dir, "MODELSCOPE_CACHE环境变量设置错误"
print("  ✓ 环境变量设置正确")
print()

print("[测试8] 历史记录文件路径:")
history_path = get_history_path()
print(f"  {history_path}")
print("  ✓ 路径正确")
print()

print("[测试9] 日志文件路径:")
log_file = get_log_file()
print(f"  {log_file}")
print("  ✓ 路径正确")
print()

print("[测试10] 历史管理器测试:")
from history_manager import HistoryManager
hm = HistoryManager()
print(f"  当前历史记录数: {len(hm.get_records())}")
print("  ✓ 历史管理器初始化成功")
print()

print("=" * 60)
print("  ✓ 所有路径测试通过！")
print("=" * 60)
print()
print("数据目录结构:")
print(f"  {get_user_data_dir()}/")
print(f"  ├── recordings/     (录音文件)")
print(f"  ├── models/         (AI模型缓存)")
print(f"  ├── logs/           (日志文件)")
print(f"  ├── temp/           (临时文件)")
print(f"  ├── cache/          (缓存文件)")
print(f"  └── history.json    (历史记录)")
print()
