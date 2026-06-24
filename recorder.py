# -*- coding: utf-8 -*-
"""
语音情绪识别系统 - 音频录制模块

本模块负责通过系统麦克风录制音频，保存为 WAV 格式文件。
采用线程化设计，录音过程在独立线程中运行，不会阻塞 GUI 界面。

主要功能：
1. 实时音频录制（支持 16kHz 单声道，与情绪识别模型要求一致）
2. 录音状态管理（录音中/已停止）
3. 实时录音时长统计
4. 实时音量检测（用于 UI 显示音量指示）
5. WAV 文件保存（16-bit PCM 格式）
6. 线程安全的状态访问

录音参数说明：
- 采样率：16000 Hz（emotion2vec 模型要求）
- 声道数：1（单声道）
- 采样位深：16-bit（paInt16）
- 缓冲区大小：1024 帧

作者：Jiawei Li
许可证：MIT License
"""

import pyaudio
import wave
import threading
import time
import os
import numpy as np


class AudioRecorder:
    """
    音频录制器类

    封装了 PyAudio 的录音功能，提供简洁的接口：
    - start_recording(output_path): 开始录音
    - stop_recording(): 停止录音并保存文件
    - is_recording(): 查询是否正在录音
    - get_duration(): 获取录音时长
    - get_volume_level(): 获取当前音量级别（0.0-1.0）

    线程安全性：
    所有公共方法都通过 _lock 互斥锁保护，
    可以安全地在 GUI 线程和录音线程之间调用。
    """

    def __init__(self):
        """
        初始化音频录制器

        设置录音参数并初始化状态变量。
        注意：PyAudio 实例不会在初始化时创建，而是在开始录音时创建，
        这样可以避免长时间占用音频设备。
        """
        # ----- 录音参数 -----
        self.FORMAT = pyaudio.paInt16    # 16-bit PCM 格式
        self.CHANNELS = 1                # 单声道
        self.RATE = 16000                # 采样率 16kHz（模型要求）
        self.CHUNK = 1024                # 每次读取的帧数

        # ----- 内部状态 -----
        self._p = None                   # PyAudio 实例
        self._stream = None              # 音频输入流
        self._frames = []                # 录音数据帧列表
        self._is_recording = False       # 录音状态标志
        self._record_thread = None       # 录音线程对象
        self._lock = threading.Lock()    # 互斥锁，保证线程安全
        self._start_time = None          # 录音开始时间戳
        self._total_duration = 0.0       # 录音总时长（秒）
        self._output_path = None         # 输出文件路径
        self._volume_level = 0.0         # 当前音量级别（0.0-1.0）

    def _record(self):
        """
        录音线程主函数（内部方法，不要直接调用）

        在独立线程中运行，循环从音频流中读取数据，
        直到 _is_recording 标志被设为 False。

        同时计算实时音量，用于 UI 上的音量指示。
        """
        try:
            # 初始化 PyAudio 并打开输入流
            self._p = pyaudio.PyAudio()
            self._stream = self._p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            # 录音主循环
            while self._is_recording:
                try:
                    # 读取一帧音频数据
                    # exception_on_overflow=False 防止缓冲区溢出时抛异常
                    data = self._stream.read(self.CHUNK, exception_on_overflow=False)
                    with self._lock:
                        self._frames.append(data)
                        # 计算 RMS 音量（均方根）
                        try:
                            # 将字节数据转换为 16 位整型数组
                            audio_data = np.frombuffer(data, dtype=np.int16)
                            if len(audio_data) > 0:
                                # 计算均方根值，代表音量大小
                                rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
                                # 归一化到 0.0-1.0 范围，乘以 5 是为了让音量显示更灵敏
                                self._volume_level = min(1.0, rms / 32768.0 * 5)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"录音读取错误: {e}")
                    with self._lock:
                        self._is_recording = False
                    break
        except Exception as e:
            print(f"录音初始化错误: {e}")
            with self._lock:
                self._is_recording = False
        finally:
            # 确保流和 PyAudio 实例被正确清理
            self._cleanup_stream()

    def _cleanup_stream(self):
        """
        清理音频流和 PyAudio 实例（内部方法）

        安全地停止并关闭音频流，终止 PyAudio 实例。
        即使部分操作失败，也会尽可能清理资源。
        """
        # 关闭音频流
        if self._stream is not None:
            try:
                if self._stream.is_active():
                    self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        # 终止 PyAudio 实例
        if self._p is not None:
            try:
                self._p.terminate()
            except Exception:
                pass
            self._p = None

    def start_recording(self, output_path='./temp_recording.wav'):
        """
        开始录音

        创建录音线程并启动。如果已经在录音中，则返回 False。

        参数：
            output_path (str): 录音保存的文件路径，默认为当前目录下的 temp_recording.wav

        返回值：
            bool: True 表示录音成功启动，False 表示录音已在进行中
        """
        with self._lock:
            if self._is_recording:
                print("录音已在进行中")
                return False
            # 重置状态
            self._frames = []
            self._is_recording = True
            self._start_time = time.time()
            self._total_duration = 0.0
            self._output_path = output_path
            self._volume_level = 0.0
        # 启动录音线程（守护线程，程序退出时自动结束）
        self._record_thread = threading.Thread(target=self._record, daemon=True)
        self._record_thread.start()
        return True

    def stop_recording(self):
        """
        停止录音并保存文件

        设置停止标志，等待录音线程结束，然后将录音数据保存为 WAV 文件。

        返回值：
            str 或 None: 成功返回保存的文件路径，失败返回 None
        """
        with self._lock:
            if not self._is_recording:
                print("当前没有进行中的录音")
                return None
            # 计算录音总时长
            self._total_duration = time.time() - self._start_time if self._start_time else 0.0
            self._is_recording = False
        # 等待录音线程结束（最多等待 5 秒，防止死锁）
        if self._record_thread is not None:
            self._record_thread.join(timeout=5.0)
            self._record_thread = None
        # 保存 WAV 文件
        output_path = self._save_wav()
        # 清理音频资源
        self._cleanup_stream()
        return output_path

    def _save_wav(self):
        """
        将录音数据保存为 WAV 文件（内部方法）

        返回值：
            str 或 None: 成功返回文件路径，失败返回 None
        """
        if not self._frames:
            print("没有录音数据可保存")
            return None
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(self._output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            # 写入 WAV 文件
            wf = wave.open(self._output_path, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(pyaudio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self._frames))
            wf.close()
            return self._output_path
        except Exception as e:
            print(f"保存WAV文件失败: {e}")
            return None

    def is_recording(self):
        """
        查询当前是否正在录音

        返回值：
            bool: True 表示正在录音，False 表示未在录音
        """
        with self._lock:
            return self._is_recording

    def get_duration(self):
        """
        获取录音时长

        如果正在录音，返回已录制的时长；
        如果已停止录音，返回上一次录音的总时长。

        返回值：
            float: 录音时长（秒）
        """
        with self._lock:
            if self._is_recording:
                if self._start_time is None:
                    return 0.0
                return time.time() - self._start_time
            else:
                return self._total_duration

    def get_volume_level(self):
        """
        获取当前录音音量级别

        返回值：
            float: 音量级别，范围 0.0（静音）到 1.0（最大音量）
        """
        with self._lock:
            return self._volume_level

    def __del__(self):
        """
        析构函数，确保资源被正确释放

        如果对象被销毁时仍在录音，会自动停止录音并清理资源。
        """
        try:
            if self._is_recording:
                self.stop_recording()
            else:
                self._cleanup_stream()
        except Exception:
            pass
