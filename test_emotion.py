import os
import sys

def test_emotion_recognition():
    print("=" * 70)
    print("语音情绪识别系统 - 自动化测试")
    print("=" * 70)
    
    recordings_dir = r"c:\Work\Voice_Mood_Detect\recordings"
    if not os.path.exists(recordings_dir):
        print(f"✗ 录音目录不存在: {recordings_dir}")
        return False
    
    test_files = [f for f in os.listdir(recordings_dir) if f.endswith('.wav')]
    if not test_files:
        print("✗ 未找到测试音频文件")
        return False
    
    print(f"\n找到 {len(test_files)} 个测试音频文件:")
    for f in test_files:
        print(f"  - {f}")
    
    print("\n" + "-" * 70)
    print("正在加载情绪识别模型...")
    print("-" * 70)
    
    from emotion_recognizer import EmotionRecognizer
    recognizer = EmotionRecognizer()
    
    if not recognizer.loaded:
        print(f"✗ 模型加载失败: {recognizer.error}")
        return False
    
    print("✓ 模型加载成功")
    
    all_passed = True
    results = []
    
    for filename in test_files:
        audio_path = os.path.join(recordings_dir, filename)
        print(f"\n{'='*70}")
        print(f"测试音频: {filename}")
        print(f"文件路径: {audio_path}")
        print("-" * 70)
        
        result = recognizer.predict(audio_path)
        
        if result.get("success", False):
            print("✓ 推理成功！")
            print(f"  主要情绪: {result['主要情绪']}")
            print(f"  置信度: {result['置信度']:.2%}")
            print(f"  焦虑分数: {result['焦虑分数']:.1f}/10")
            print(f"  情绪等级: {result['情绪等级']}")
            print(f"\n  情绪概率分布:")
            for emotion, prob in sorted(result['所有情绪概率'].items(), key=lambda x: -x[1]):
                bar = "█" * int(prob * 30)
                print(f"    {emotion:4s}: {prob:6.2%} {bar}")
            results.append((filename, True, result))
        else:
            print(f"✗ 推理失败: {result.get('error', '未知错误')}")
            all_passed = False
            results.append((filename, False, result))
    
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n总测试数: {total}")
    print(f"通过: {passed}")
    print(f"失败: {total - passed}")
    
    for filename, ok, result in results:
        status = "✓ 通过" if ok else "✗ 失败"
        if ok:
            print(f"  {status} - {filename} -> {result['情绪等级']} (分数: {result['焦虑分数']:.1f})")
        else:
            print(f"  {status} - {filename} -> 错误: {result.get('error', '未知')}")
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ 所有测试通过！情绪识别功能正常。")
    else:
        print("✗ 部分测试失败，请检查问题。")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = test_emotion_recognition()
    sys.exit(0 if success else 1)
