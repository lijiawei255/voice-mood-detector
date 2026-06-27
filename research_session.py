# -*- coding: utf-8 -*-
"""
语音情绪识别系统 - 科研评估会话模块 (P1)

本模块实现科研评估模式的完整流程编排：
1. 统一提示语录音（建议 10-30 秒）
2. 音频质量门控（有效语音、音量、噪声、时长）
3. 多次采样综合评估（默认 3 段）
4. 双模型规模敏感性验证
5. 多次采样一致性分析

使用方式：
    session = ResearchSession(recorder, recognizer)
    session.run_session(prompt_text="请用平稳的语气描述今天的感受",
                        n_samples=3,
                        progress_callback=lambda msg, progress: print(msg),
                        finished_callback=lambda result: print(result))

作者：Jiawei Li
许可证：GPL v3
"""

import os
import time
import logging
import tempfile
import threading
from datetime import datetime

from audio_quality import AudioQualityAnalyzer, AUDIO_QUALITY_THRESHOLDS
from reliability import evaluate_multi_sample_reliability, get_reliability_level
from version import APP_VERSION, ALGORITHM_VERSION

logger = logging.getLogger(__name__)


class ResearchSession:
    """
    科研评估会话管理器

    封装科研模式的完整录音-分析流程，提供同步与异步两种调用方式。
    """

    # 默认配置
    DEFAULT_N_SAMPLES = 3
    RESEARCH_MIN_DURATION = 10
    RESEARCH_MAX_DURATION = 30

    def __init__(self, recorder, recognizer):
        """
        初始化科研会话

        参数：
            recorder (AudioRecorder): 录音器实例
            recognizer (EmotionRecognizer): 情绪识别器实例
        """
        self.recorder = recorder
        self.recognizer = recognizer
        self.cancelled = False
        self._stop_sample = False
        self._lock = threading.Lock()

    def cancel(self):
        """取消当前会话"""
        with self._lock:
            self.cancelled = True
        try:
            if self.recorder.is_recording():
                self.recorder.stop_recording()
        except Exception as e:
            logger.warning(f"取消会话时停止录音失败: {e}")

    def _is_cancelled(self):
        with self._lock:
            return self.cancelled

    def request_stop_current_sample(self):
        """
        请求停止当前段录音（不取消整个会话）

        与 cancel() 的区别：仅结束当前正在录制的样本，会话继续进入
        质量门控与下一段采样。用于让用户手动结束某段录音。

        线程安全：由主线程调用，record_sample 在后台线程轮询该标志。
        """
        with self._lock:
            self._stop_sample = True
        # 立即停止录音，让 record_sample 的等待循环尽快退出
        try:
            if self.recorder.is_recording():
                self.recorder.stop_recording()
        except Exception as e:
            logger.warning(f"手动停止当前段录音失败: {e}")

    def _is_stop_sample_requested(self):
        with self._lock:
            return self._stop_sample

    def quality_gate(self, audio_path, is_research=True):
        """
        对录音进行质量门控检查

        仅当出现「致命」问题（时长不足/过长、爆音）时不通过；噪声/语音比例/
        音量偏低仅作为 warning 返回，不阻断科研流程（对真实麦克风录音保持
        宽容）。

        参数：
            audio_path (str): 音频文件路径
            is_research (bool): 是否使用科研模式更严格阈值

        返回值：
            dict: {"passed": bool, "quality": dict, "reason": str, "warnings": list}
        """
        analyzer = AudioQualityAnalyzer()
        quality = analyzer.analyze(audio_path)
        if not quality:
            return {"passed": False, "quality": {}, "reason": "无法分析音频质量", "warnings": []}

        # 致命问题（来自 analyzer 的 issues）+ 科研模式时长约束
        fatal = list(quality.get("issues", []))
        warnings = list(quality.get("warnings", []))
        duration = quality.get("duration", 0)
        thresholds = AUDIO_QUALITY_THRESHOLDS

        if is_research:
            if duration < thresholds.get("research_min_duration", 3):
                fatal.append(f"科研模式录音时长不足（需 ≥{thresholds.get('research_min_duration')} 秒）")
            if duration > thresholds.get("research_max_duration", 30):
                fatal.append(f"科研模式录音时长过长（需 ≤{thresholds.get('research_max_duration')} 秒）")
        else:
            if duration < thresholds.get("min_duration", 1):
                fatal.append("录音时长过短")

        if fatal:
            return {
                "passed": False,
                "quality": quality,
                "reason": "；".join(fatal),
                "warnings": warnings,
            }

        # 无致命问题即放行（warnings 仅作提示，不影响 passed）
        return {
            "passed": True,
            "quality": quality,
            "reason": "",
            "warnings": warnings,
        }

    def record_sample(self, output_path, max_duration=30, progress_callback=None):
        """
        录制一段样本

        参数：
            output_path (str): 输出文件路径
            max_duration (int): 最大录音时长
            progress_callback (callable): 进度回调，接收 (message, elapsed_seconds)

        返回值：
            bool: 是否成功录制
        """
        success = self.recorder.start_recording(output_path)
        if not success:
            return False

        elapsed = 0.0
        while (elapsed < max_duration
               and self.recorder.is_recording()
               and not self._is_cancelled()
               and not self._is_stop_sample_requested()):
            time.sleep(0.2)
            elapsed += 0.2
            if progress_callback and int(elapsed * 5) % 5 == 0:
                try:
                    progress_callback(f"录音中... {elapsed:.1f} 秒", elapsed)
                except Exception:
                    pass

        if self.recorder.is_recording():
            self.recorder.stop_recording()

        # 用户请求仅停止当前段（非取消整个会话）：重置标志，视为本段成功录制，
        # 让会话继续进入质量门控（过短则由 quality_gate 优雅反馈）
        if self._is_stop_sample_requested():
            with self._lock:
                self._stop_sample = False
            return os.path.exists(output_path) and not self._is_cancelled()

        return not self._is_cancelled() and os.path.exists(output_path)

    def analyze_sample(self, audio_path, use_dual_model=False):
        """
        分析单段样本

        参数：
            audio_path (str): 音频文件路径
            use_dual_model (bool): 是否使用双模型验证

        返回值：
            dict: 分析结果
        """
        if use_dual_model and self.recognizer.model_name in ["emotion2vec_plus_base", "emotion2vec_plus_large"]:
            return self.recognizer.predict_dual_model(audio_path)
        return self.recognizer.predict(audio_path, is_research_mode=True)

    def run_session(self, prompt_text=None, n_samples=DEFAULT_N_SAMPLES,
                    use_dual_model=True, progress_callback=None,
                    finished_callback=None):
        """
        运行完整科研评估会话（同步阻塞方式）

        参数：
            prompt_text (str): 统一提示语
            n_samples (int): 采样段数（默认 3）
            use_dual_model (bool): 是否启用双模型验证
            progress_callback (callable): 进度回调，接收 (phase, message, progress_percent)
            finished_callback (callable): 完成回调，接收最终 result dict

        返回值：
            dict: 会话结果
        """
        self.cancelled = False
        results = []
        sample_paths = []

        def _report(phase, message, progress):
            if progress_callback:
                try:
                    progress_callback(phase, message, progress)
                except Exception:
                    pass

        try:
            _report("prepare", "科研评估模式启动", 5)

            # 1. 多次采样录音
            if prompt_text:
                _report("prompt", f"提示语：{prompt_text}", 8)

            for i in range(n_samples):
                if self._is_cancelled():
                    break

                sample_path = tempfile.mktemp(suffix=f"_research_sample_{i+1}.wav")
                sample_paths.append(sample_path)

                _report(
                    "record",
                    f"第 {i+1}/{n_samples} 段录音：请按提示语朗读（建议 {self.RESEARCH_MIN_DURATION}-{self.RESEARCH_MAX_DURATION} 秒，最短 {int(AUDIO_QUALITY_THRESHOLDS.get('research_min_duration', 3.0))} 秒）",
                    10 + (i + 1) * 20 // n_samples,
                )

                ok = self.record_sample(
                    sample_path,
                    max_duration=self.RESEARCH_MAX_DURATION,
                    progress_callback=lambda msg, sec: _report("record", f"第 {i+1}/{n_samples} 段 {msg}", 10 + (i + 1) * 20 // n_samples),
                )
                if not ok:
                    error_result = {"success": False, "error": f"第 {i+1} 段录音失败或被取消", "phase": "recording"}
                    if finished_callback:
                        finished_callback(error_result)
                    return error_result

                # 质量门控
                _report("quality", f"第 {i+1}/{n_samples} 段：正在检查录音质量...", 12 + (i + 1) * 22 // n_samples)
                gate = self.quality_gate(sample_path, is_research=True)
                if not gate["passed"]:
                    error_result = {
                        "success": False,
                        "error": f"第 {i+1} 段录音未通过质量门控：{gate['reason']}",
                        "phase": "quality_gate",
                        "quality": gate.get("quality", {}),
                    }
                    if finished_callback:
                        finished_callback(error_result)
                    return error_result

                # 分析
                _report("analyze", f"第 {i+1}/{n_samples} 段：正在分析...", 15 + (i + 1) * 22 // n_samples)
                sample_result = self.analyze_sample(sample_path, use_dual_model=use_dual_model)
                if not sample_result.get("success"):
                    error_result = {
                        "success": False,
                        "error": f"第 {i+1} 段分析失败：{sample_result.get('error')}",
                        "phase": "analysis",
                    }
                    if finished_callback:
                        finished_callback(error_result)
                    return error_result

                # 将音频路径补充到结果，便于后续导出或保存
                sample_result["audio_file"] = sample_path
                sample_result["audio_quality"] = gate.get("quality", sample_result.get("audio_quality", {}))
                results.append(sample_result)

            if self._is_cancelled():
                error_result = {"success": False, "error": "会话已取消", "phase": "cancelled"}
                if finished_callback:
                    finished_callback(error_result)
                return error_result

            # 3. 多次采样综合评估
            _report("reliability", "正在计算多次采样一致性...", 80)
            multi_sample = evaluate_multi_sample_reliability(results)

            # 4. 综合可靠性（音频质量 + 置信度 + 多次采样一致性）
            avg_quality_score = 0.0
            quality_count = 0
            for r in results:
                aq = r.get("audio_quality", {})
                if isinstance(aq, dict) and "quality_score" in aq:
                    avg_quality_score += aq["quality_score"]
                    quality_count += 1
            avg_quality_score = avg_quality_score / quality_count if quality_count > 0 else 0.5

            overall_reliability = get_reliability_level({
                "audio_quality": {"quality_score": avg_quality_score},
                "confidence": results[0].get("置信度", 0.5) if results else 0.5,
                "consistency": multi_sample.get("consistency"),
            })

            # 5. 组装最终报告
            final_result = {
                "success": True,
                "is_research_session": True,
                "n_samples": len(results),
                "prompt_text": prompt_text or "",
                "sample_results": results,
                "session_emotion": multi_sample.get("session_emotion", ""),
                "session_stability_mean": multi_sample.get("stability_mean", 0.0),
                "session_stability_std": multi_sample.get("stability_std", 0.0),
                "session_valence_mean": multi_sample.get("valence_mean", 0.0),
                "session_valence_std": multi_sample.get("valence_std", 0.0),
                "session_arousal_mean": multi_sample.get("arousal_mean", 0.0),
                "session_arousal_std": multi_sample.get("arousal_std", 0.0),
                "emotion_consistency": multi_sample.get("emotion_consistency", 0.0),
                "consistency": multi_sample.get("consistency", 0.0),
                "model_agreement": results[0].get("model_agreement") if results and use_dual_model else None,
                "result_reliability": multi_sample.get("reliability", "低"),
                "overall_reliability": overall_reliability,
                "assessment_reliability": overall_reliability,
                "sample_reliability": multi_sample.get("reliability", "低"),
                "model_agreement_note": results[0].get("model_agreement_note", "") if results else "",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "algorithm_version": ALGORITHM_VERSION,
                "app_version": APP_VERSION,
                "is_research_mode": True,
            }

            # 汇总音频质量
            final_result["audio_quality"] = {
                "avg_quality_score": round(avg_quality_score, 4),
                "samples": [r.get("audio_quality", {}) for r in results],
            }

            _report("done", "科研评估完成", 100)
            if finished_callback:
                finished_callback(final_result)
            return final_result

        except Exception as e:
            logger.error(f"科研会话运行失败: {e}", exc_info=True)
            error_result = {"success": False, "error": f"科研会话异常: {e}", "phase": "unknown"}
            if finished_callback:
                finished_callback(error_result)
            return error_result

    def run_async(self, *args, **kwargs):
        """
        异步运行科研评估会话

        参数与 run_session 相同，但在后台线程执行。

        返回值：
            threading.Thread: 启动的后台线程
        """
        thread = threading.Thread(target=self.run_session, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread
