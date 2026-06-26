import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("语音情绪识别系统 - 录音流程自动化测试")
print("=" * 70)
print()

print("[测试1] 测试 AudioRecorder 类基本功能...")
try:
    from recorder import AudioRecorder
    recorder = AudioRecorder()
    print("  ✓ AudioRecorder 初始化成功")

    temp_wav = os.path.join(os.path.dirname(__file__), "portable_data", "temp", "test_recording.wav")
    os.makedirs(os.path.dirname(temp_wav), exist_ok=True)

    print("  测试启动录音...")
    success = recorder.start_recording(temp_wav)
    if success:
        print("  ✓ 录音启动成功")
        print("  等待2秒...")
        time.sleep(2)

        is_rec = recorder.is_recording()
        print(f"  录音状态: {'正在录音' if is_rec else '未录音'}")

        duration = recorder.get_duration()
        print(f"  当前时长: {duration:.2f}秒")

        print("  测试停止录音...")
        output_path = recorder.stop_recording()
        if output_path and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"  ✓ 录音停止成功，文件大小: {file_size} 字节")

            final_duration = recorder.get_duration()
            print(f"  ✓ 最终录音时长: {final_duration:.2f}秒")

            if file_size > 1024:
                print("  ✓ 录音文件有效")
                try:
                    os.remove(output_path)
                    print("  ✓ 测试文件已清理")
                except:
                    pass
            else:
                print("  ⚠ 录音文件过小，可能有问题")
        else:
            print("  ✗ 录音停止失败")
    else:
        print("  ✗ 录音启动失败")

except Exception as e:
    print(f"  ✗ 测试异常: {e}")
    import traceback
    traceback.print_exc()

print()
print("[测试2] 测试模块导入...")
try:
    os.environ['FUNASR_AUTO_INSTALL'] = '0'
    os.environ['FUNASR_INSTALL_DEP'] = '0'
    os.environ['MODELSCOPE_AUTO_INSTALL_DEP'] = '0'

    from app_paths import get_app_dir, get_user_data_dir, get_recordings_dir
    print(f"  ✓ 路径模块导入成功")
    print(f"    程序目录: {get_app_dir()}")
    print(f"    数据目录: {get_user_data_dir()}")
    print(f"    录音目录: {get_recordings_dir()}")

    from emotion_recognizer import EmotionRecognizer
    print(f"  ✓ 情绪识别模块导入成功")

    from PyQt5.QtWidgets import QApplication
    from gui import MainWindow
    print(f"  ✓ GUI模块导入成功")

except Exception as e:
    print(f"  ✗ 导入异常: {e}")
    import traceback
    traceback.print_exc()

print()
print("[测试3] 测试现有录音文件分析...")
try:
    recordings_dir = get_recordings_dir()
    test_files = [f for f in os.listdir(recordings_dir) if f.endswith('.wav')] if os.path.exists(recordings_dir) else []
    if test_files:
        print(f"  找到 {len(test_files)} 个录音文件")
        for f in test_files[:3]:
            path = os.path.join(recordings_dir, f)
            size = os.path.getsize(path)
            print(f"    - {f} ({size/1024:.1f} KB)")
    else:
        print("  未找到现有录音文件（这是正常的，首次运行后会有）")
except Exception as e:
    print(f"  ⚠ 检查录音文件时出错: {e}")

print()
print("=" * 70)
print("测试完成！")
print("=" * 70)
print()
print("接下来请运行: python main.py")
print("然后测试以下流程：")
print("1. 等待模型加载完成")
print("2. 点击「开始录音」，观察日志是否有「✅ 录音已开始」")
print("3. 等待几秒，观察时长是否更新")
print("4. 点击「停止录音」，观察日志输出")
