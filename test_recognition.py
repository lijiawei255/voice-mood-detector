import os
import sys

os.environ['FUNASR_AUTO_INSTALL'] = '0'
os.environ['FUNASR_INSTALL_DEP'] = '0'
os.environ['MODELSCOPE_AUTO_INSTALL_DEP'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app_paths import setup_modelscope_cache
setup_modelscope_cache()

print("正在加载模型...")
from emotion_recognizer import EmotionRecognizer

recognizer = EmotionRecognizer()
recognizer.load_model()

import time
while not recognizer.is_ready():
    time.sleep(0.5)

print("模型加载完成，开始测试\n")

test_dir = os.path.dirname(os.path.abspath(__file__))
recordings_dir = os.path.join(test_dir, "portable_data", "recordings")
test_files = []
if os.path.exists(recordings_dir):
    for f in os.listdir(recordings_dir):
        if f.endswith('.wav'):
            test_files.append(os.path.join(recordings_dir, f))

for f in os.listdir(test_dir):
    if f.endswith('.wav'):
        test_files.append(os.path.join(test_dir, f))

test_files = list(set(test_files))
print(f"找到 {len(test_files)} 个测试文件\n")

for audio_path in test_files:
    print("=" * 60)
    print(f"文件: {os.path.basename(audio_path)}")
    result = recognizer.predict(audio_path)
    if result["success"]:
        print(f"主要情绪: {result['主要情绪']} (置信度: {result['置信度']*100:.1f}%)")
        print(f"情绪状态: {result['情绪状态等级']} (稳定度分数: {result['情绪稳定度分数']}/10)")
        if result.get("混合情绪"):
            mixed = ", ".join([f"{e}({p*100:.1f}%)" for e, p in result["混合情绪"]])
            print(f"混合情绪: {mixed}")
        print("情绪分布:")
        for emo, prob in sorted(result["所有情绪概率"].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 30)
            print(f"  {emo:4s}: {prob*100:5.1f}% {bar}")
        print(f"建议: {result['调节建议']}")
    else:
        print(f"识别失败: {result.get('error')}")
    print()
