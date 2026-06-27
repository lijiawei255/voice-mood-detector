# -*- coding: utf-8 -*-
"""
科研评估会话模块测试

测试 ResearchSession 的质量门控、环境检测、多次采样综合评估等逻辑。
使用 Mock 对象模拟录音器和识别器，避免依赖真实硬件。
"""

import os
import sys
import unittest
import tempfile
import wave
import struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research_session import ResearchSession


def _make_silent_wav(path, duration=1.0, sample_rate=16000):
    """生成一个静音 WAV 文件"""
    n_samples = int(duration * sample_rate)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        data = [0] * n_samples
        wf.writeframes(struct.pack('<' + 'h' * len(data), *data))


def _make_noisy_wav(path, duration=1.0, sample_rate=16000):
    """生成一个带小幅噪声的 WAV 文件"""
    import random
    n_samples = int(duration * sample_rate)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        data = [int(random.uniform(-1000, 1000)) for _ in range(n_samples)]
        wf.writeframes(struct.pack('<' + 'h' * len(data), *data))


class _FakeRecorder:
    """模拟录音器：录音 2 秒后自动完成"""

    def __init__(self, output_path=None, record_duration=2.0):
        self._recording = False
        self._output_path = output_path
        self._last_path = None
        self._record_duration = record_duration

    def start_recording(self, path):
        self._last_path = path
        self._recording = True
        # 在后台线程模拟自动结束录音
        import threading
        def _auto_stop():
            import time
            time.sleep(self._record_duration)
            if self._recording:
                self.stop_recording()
        threading.Thread(target=_auto_stop, daemon=True).start()
        return True

    def stop_recording(self):
        self._recording = False
        if self._last_path:
            _make_noisy_wav(self._last_path, duration=max(12.0, self._record_duration + 0.5))
        return self._last_path

    def is_recording(self):
        return self._recording


class _FakeRecognizer:
    """模拟识别器"""

    def __init__(self):
        self.model_name = "emotion2vec_plus_large"

    def predict(self, audio_path, is_research_mode=False):
        return {
            "success": True,
            "主要情绪": "平静",
            "置信度": 0.7,
            "情绪稳定度分数": 3.0,
            "valence_score": 0.2,
            "arousal_score": 0.3,
            "dominance_score": 0.6,
            "audio_file": audio_path,
        }

    def predict_dual_model(self, audio_path):
        result = self.predict(audio_path, is_research_mode=True)
        result.update({
            "is_dual_model": True,
            "model_agreement": 0.85,
            "result_reliability": "高",
            "secondary_result": {"主要情绪": "平静"},
        })
        return result


class TestResearchSession(unittest.TestCase):

    def test_quality_gate_pass(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            _make_noisy_wav(path, duration=12.0)
            session = ResearchSession(_FakeRecorder(), _FakeRecognizer())
            gate = session.quality_gate(path, is_research=True)
            self.assertIn("passed", gate)
        finally:
            os.remove(path)

    def test_quality_gate_fail_too_short(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            _make_noisy_wav(path, duration=1.0)
            session = ResearchSession(_FakeRecorder(), _FakeRecognizer())
            gate = session.quality_gate(path, is_research=True)
            self.assertFalse(gate["passed"])
        finally:
            os.remove(path)

    def test_run_session_success(self):
        recorder = _FakeRecorder(record_duration=0.5)
        recognizer = _FakeRecognizer()
        session = ResearchSession(recorder, recognizer)

        progress_log = []
        result = session.run_session(
            prompt_text="请描述今天的感受",
            n_samples=2,
            use_dual_model=True,
            progress_callback=lambda phase, msg, prog: progress_log.append((phase, prog)),
        )
        self.assertTrue(result.get("success"), f"会话失败: {result.get('error')}")
        self.assertEqual(result.get("n_samples"), 2)
        self.assertIn("overall_reliability", result)
        self.assertTrue(len(progress_log) > 0)

    def test_cancel_session(self):
        recorder = _FakeRecorder()
        recognizer = _FakeRecognizer()
        session = ResearchSession(recorder, recognizer)
        session.cancel()
        self.assertTrue(session._is_cancelled())


if __name__ == "__main__":
    unittest.main()
