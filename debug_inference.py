import os
import sys
import pprint

os.environ['MODELSCOPE_CACHE'] = os.path.join(os.path.expanduser('~'), '.cache', 'modelscope')

print("=" * 60)
print("调试模型推理结果格式")
print("=" * 60)

from emotion_recognizer import EmotionRecognizer

recognizer = EmotionRecognizer()
if not recognizer.loaded:
    print(f"模型加载失败: {recognizer.error}")
    sys.exit(1)

test_audio = r"c:\Work\Voice_Mood_Detect\recordings\recording_20260624_172827.wav"
print(f"\n测试音频: {test_audio}")
print("文件存在:", os.path.exists(test_audio))

print("\n正在执行推理...")
import torch
with torch.no_grad():
    raw_result = recognizer.model(test_audio, granularity="utterance")

print("\n" + "=" * 60)
print("原始返回结果:")
print("=" * 60)
pprint.pprint(raw_result)
print("=" * 60)

print("\n结果键:", list(raw_result.keys()) if isinstance(raw_result, dict) else "不是字典")
if isinstance(raw_result, dict) and "scores" in raw_result:
    scores = raw_result["scores"]
    print(f"\nscores类型: {type(scores)}")
    print(f"scores内容: {scores}")
    if isinstance(scores, list):
        print(f"scores长度: {len(scores)}")
        if len(scores) > 0:
            print(f"scores[0]类型: {type(scores[0])}")
