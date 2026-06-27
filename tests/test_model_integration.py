# -*- coding: utf-8 -*-
"""
模型集成测试

使用 portable_data/recordings/ 下的 WAV 文件进行端到端推理测试：
- predict() 返回所有必需字段
- 返回值类型正确
- 数值范围约束验证
- 批量音频文件无崩溃测试

注意：模型可能未下载，使用 @unittest.skipIf 检查。

作者：Jiawei Li
许可证：GPL v3
"""

import glob
import os
import sys
import unittest
from unittest.mock import MagicMock

# 在导入前设置环境变量，避免触发模型自动下载
os.environ["FUNASR_AUTO_INSTALL"] = "0"
os.environ["FUNASR_INSTALL_DEP"] = "0"
os.environ["MODELSCOPE_AUTO_INSTALL_DEP"] = "0"

# 如果 torch 未安装，注入 mock 以确保模块能正常导入（实际推理会因模型未加载而 skip）
_torch_available = True
try:
    import torch  # noqa: F401
except ImportError:
    _torch_available = False
    sys.modules["torch"] = MagicMock()

from app_paths import is_model_downloaded, load_model_config


def _get_wav_files():
    """获取 portable_data/recordings/ 目录下的 WAV 文件（最多取 3 个）"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    recordings_dir = os.path.join(project_root, "portable_data", "recordings")
    if not os.path.exists(recordings_dir):
        return []
    wav_files = sorted(glob.glob(os.path.join(recordings_dir, "*.wav")))
    return wav_files[:3]


# 判断当前配置的模型是否已下载，且 torch 可用
_current_model = load_model_config()
_current_model_downloaded = is_model_downloaded(_current_model) and _torch_available
_wav_files = _get_wav_files()


class TestPredictFields(unittest.TestCase):
    """predict() 返回字段完整性测试"""

    @unittest.skipIf(not _current_model_downloaded, "未检测到已下载的模型，跳过集成测试")
    def setUp(self):
        from emotion_recognizer import EmotionRecognizer
        self.recognizer = EmotionRecognizer()
        # 同步加载模型（阻塞直到完成）
        import threading
        loaded_event = threading.Event()
        self._load_error = None

        def on_loaded(success, error):
            if not success:
                self._load_error = error
            loaded_event.set()

        self.recognizer.load_model(callback=on_loaded)
        loaded_event.wait(timeout=120)  # 最多等 120 秒
        if not self.recognizer.is_ready():
            self.skipTest(f"模型加载失败: {self._load_error or '超时'}")

    @unittest.skipIf(not _current_model_downloaded, "未检测到已下载的模型，跳过集成测试")
    @unittest.skipIf(len(_wav_files) == 0, "未找到可用的 WAV 测试文件")
    def test_predict_returns_all_fields(self):
        """predict() 返回包含所有必需字段的字典"""
        result = self.recognizer.predict(_wav_files[0])
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("success", False),
                        f"predict() 失败: {result.get('error', '未知错误')}")

        # 当前 predict() 返回的字段
        required_fields = [
            "主要情绪", "置信度",
            "所有情绪概率",
            "完整概率_8类",
            "原始模型输出",
            "情绪稳定度分数", "情绪状态等级", "等级颜色",
            "稳定度分项",
            "valence_score", "arousal_score", "dominance_score",
            "negative_load", "emotional_uncertainty", "estimation_note",
            "调节建议", "混合情绪",
            "复合情绪", "复合情绪详情",
            "情绪分析摘要",
        ]
        for field in required_fields:
            self.assertIn(field, result,
                          f"返回结果缺少字段: '{field}'")

    @unittest.skipIf(not _current_model_downloaded, "未检测到已下载的模型，跳过集成测试")
    @unittest.skipIf(len(_wav_files) == 0, "未找到可用的 WAV 测试文件")
    def test_predict_field_types(self):
        """各字段的返回值类型正确"""
        result = self.recognizer.predict(_wav_files[0])
        if not result.get("success"):
            self.skipTest(f"predict() 失败: {result.get('error')}")

        # 类型断言
        self.assertIsInstance(result["主要情绪"], str)
        self.assertIsInstance(result["置信度"], float)
        self.assertIsInstance(result["所有情绪概率"], dict)
        self.assertIsInstance(result["完整概率_8类"], dict)
        self.assertIsInstance(result["原始模型输出"], dict)
        self.assertIsInstance(result["情绪稳定度分数"], float)
        self.assertIsInstance(result["情绪状态等级"], str)
        self.assertIsInstance(result["等级颜色"], str)
        self.assertIsInstance(result["稳定度分项"], dict)
        self.assertIsInstance(result["valence_score"], float)
        self.assertIsInstance(result["arousal_score"], float)
        self.assertIsInstance(result["dominance_score"], float)
        self.assertIsInstance(result["negative_load"], float)
        self.assertIsInstance(result["emotional_uncertainty"], float)
        self.assertIsInstance(result["estimation_note"], str)
        self.assertIsInstance(result["调节建议"], str)
        self.assertIsInstance(result["混合情绪"], list)
        self.assertIsInstance(result["复合情绪"], str)
        # 复合情绪详情可能为 dict 或 None
        self.assertTrue(
            result["复合情绪详情"] is None or isinstance(result["复合情绪详情"], dict)
        )
        self.assertIsInstance(result["情绪分析摘要"], str)

    @unittest.skipIf(not _current_model_downloaded, "未检测到已下载的模型，跳过集成测试")
    @unittest.skipIf(len(_wav_files) == 0, "未找到可用的 WAV 测试文件")
    def test_predict_value_ranges(self):
        """数值范围约束验证"""
        result = self.recognizer.predict(_wav_files[0])
        if not result.get("success"):
            self.skipTest(f"predict() 失败: {result.get('error')}")

        # 置信度 0-1
        confidence = result["置信度"]
        self.assertGreaterEqual(confidence, 0.0,
                                f"置信度 {confidence} < 0")
        self.assertLessEqual(confidence, 1.0,
                             f"置信度 {confidence} > 1")

        # 稳定度 0-10
        stability = result["情绪稳定度分数"]
        self.assertGreaterEqual(stability, 0.0,
                                f"稳定度 {stability} < 0")
        self.assertLessEqual(stability, 10.0,
                             f"稳定度 {stability} > 10")

        # VAD 分数范围
        self.assertGreaterEqual(result["valence_score"], -1.0)
        self.assertLessEqual(result["valence_score"], 1.0)
        self.assertGreaterEqual(result["arousal_score"], 0.0)
        self.assertLessEqual(result["arousal_score"], 1.0)
        self.assertGreaterEqual(result["dominance_score"], 0.0)
        self.assertLessEqual(result["dominance_score"], 1.0)
        self.assertGreaterEqual(result["negative_load"], 0.0)
        self.assertLessEqual(result["negative_load"], 1.0)
        self.assertGreaterEqual(result["emotional_uncertainty"], 0.0)
        self.assertLessEqual(result["emotional_uncertainty"], 1.0)

        # 8 类概率之和应接近 1
        probs_8 = result["完整概率_8类"]
        self.assertEqual(set(probs_8.keys()), set(result["所有情绪概率"].keys()) | {"其他"})
        total_8 = sum(probs_8.values())
        self.assertAlmostEqual(total_8, 1.0, places=1)


class TestBatchPredictNoCrash(unittest.TestCase):
    """批量测试多个音频文件，确保无崩溃"""

    @unittest.skipIf(not _current_model_downloaded, "未检测到已下载的模型，跳过集成测试")
    @unittest.skipIf(len(_wav_files) == 0, "未找到可用的 WAV 测试文件")
    def setUp(self):
        from emotion_recognizer import EmotionRecognizer
        self.recognizer = EmotionRecognizer()
        import threading
        loaded_event = threading.Event()
        self._load_error = None

        def on_loaded(success, error):
            if not success:
                self._load_error = error
            loaded_event.set()

        self.recognizer.load_model(callback=on_loaded)
        loaded_event.wait(timeout=120)
        if not self.recognizer.is_ready():
            self.skipTest(f"模型加载失败: {self._load_error or '超时'}")

    @unittest.skipIf(not _current_model_downloaded, "未检测到已下载的模型，跳过集成测试")
    @unittest.skipIf(len(_wav_files) == 0, "未找到可用的 WAV 测试文件")
    def test_batch_predict_no_crash(self):
        """批量测试：每个文件都不崩溃，返回字典"""
        for wav_path in _wav_files:
            with self.subTest(wav=os.path.basename(wav_path)):
                result = self.recognizer.predict(wav_path)
                self.assertIsInstance(result, dict,
                                      f"{wav_path} 返回非字典类型")
                # success 字段必须存在
                self.assertIn("success", result,
                              f"{wav_path} 缺少 'success' 字段")
                if result["success"]:
                    # 成功时应包含主要情绪
                    self.assertIn("主要情绪", result)


if __name__ == "__main__":
    unittest.main()
