# -*- coding: utf-8 -*-
"""
路径功能测试

验证 app_paths 模块提供的便携模式路径管理功能：
- 用户数据目录、录音目录、日志目录等路径正确存在
- ModelScope 缓存环境变量设置正确
- HistoryManager 能正常初始化
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app_paths import (
    get_user_data_dir,
    get_recordings_dir,
    get_logs_dir,
    get_temp_dir,
    get_cache_dir,
    get_history_path,
    get_model_cache_dir,
    get_log_file,
    setup_modelscope_cache,
)


class TestAppPaths(unittest.TestCase):
    """路径管理单元测试"""

    def test_user_data_dir_exists(self):
        self.assertTrue(os.path.isdir(get_user_data_dir()))

    def test_recordings_dir_exists(self):
        self.assertTrue(os.path.isdir(get_recordings_dir()))

    def test_logs_dir_exists(self):
        self.assertTrue(os.path.isdir(get_logs_dir()))

    def test_temp_dir_exists(self):
        self.assertTrue(os.path.isdir(get_temp_dir()))

    def test_cache_dir_exists(self):
        self.assertTrue(os.path.isdir(get_cache_dir()))

    def test_model_cache_dir_exists(self):
        self.assertTrue(os.path.isdir(get_model_cache_dir()))

    def test_modelscope_cache_env(self):
        setup_modelscope_cache()
        self.assertEqual(os.environ.get('MODELSCOPE_CACHE'), get_model_cache_dir())

    def test_history_path_is_string(self):
        self.assertIsInstance(get_history_path(), str)

    def test_log_file_is_string(self):
        self.assertIsInstance(get_log_file(), str)

    def test_history_manager_init(self):
        from history_manager import HistoryManager
        hm = HistoryManager()
        self.assertIsInstance(hm.get_records(), list)


if __name__ == "__main__":
    unittest.main()
