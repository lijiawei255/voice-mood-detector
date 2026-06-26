import os
import sys

os.environ['FUNASR_AUTO_INSTALL'] = '0'
os.environ['FUNASR_INSTALL_DEP'] = '0'
os.environ['MODELSCOPE_AUTO_INSTALL_DEP'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("情绪识别调试脚本")
print("=" * 70)

from app_paths import setup_modelscope_cache
setup_modelscope_cache()

print("\n正在加载模型...")
from funasr import AutoModel
model = AutoModel(model="iic/emotion2vec_plus_large", disable_pbar=True, disable_log=True)
print("模型加载完成！")

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
print(f"\n找到 {len(test_files)} 个测试音频文件")

if not test_files:
    print("未找到测试音频文件，请先录音后再测试")
else:
    for audio_path in test_files:
        print(f"\n{'='*60}")
        print(f"测试文件: {os.path.basename(audio_path)}")
        file_size = os.path.getsize(audio_path)
        print(f"文件大小: {file_size/1024:.1f} KB")

        try:
            result = model.generate(audio_path, granularity="utterance", extract_embedding=False)
            print(f"\n原始输出类型: {type(result)}")
            if isinstance(result, list):
                print(f"列表长度: {len(result)}")
                if len(result) > 0:
                    res = result[0]
                    print(f"\n结果项类型: {type(res)}")
                    if isinstance(res, dict):
                        print(f"结果键: {list(res.keys())}")
                        if 'labels' in res:
                            print(f"\n标签({len(res['labels'])}):")
                            for i, lab in enumerate(res['labels']):
                                print(f"  {i}: {repr(lab)}")
                        if 'scores' in res:
                            print(f"\n分数({len(res['scores'])}):")
                            scores = res['scores']
                            for i, s in enumerate(scores):
                                print(f"  {i}: {s}")
                        if 'labels' in res and 'scores' in res:
                            labels = res['labels']
                            scores = res['scores']
                            print(f"\n情绪概率分布:")
                            pairs = list(zip(labels, scores))
                            pairs.sort(key=lambda x: -x[1] if hasattr(x[1], '__float__') else -float(x[1]))
                            for lab, score in pairs:
                                try:
                                    s = float(score) * 100
                                    print(f"  {lab}: {s:.2f}%")
                                except:
                                    print(f"  {lab}: {score}")
                    else:
                        print(f"结果内容: {repr(res)[:500]}")
            else:
                print(f"原始输出: {repr(result)[:1000]}")
        except Exception as e:
            print(f"推理出错: {e}")
            import traceback
            traceback.print_exc()

print("\n" + "=" * 70)
print("调试完成")
