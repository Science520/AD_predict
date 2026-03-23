#!/usr/bin/env python3
"""
FunASR 在 SeniorTalk 测试集上的评估
"""

import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob
from pypinyin import pinyin, Style
import re
import jiwer
from datetime import datetime

# 配置
NUM_SAMPLES = 100
OUTPUT_DIR = "/data/AD_predict/experiments/funasr_eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("FunASR SenseVoice 评估")
print("="*80)

# 评估指标函数
def clean(text):
    return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', str(text)).strip()

def calc_cer(ref, hyp):
    r, h = clean(ref), clean(hyp)
    return jiwer.cer(r, h) if r else 1.0

def calc_wer(ref, hyp):
    r, h = clean(ref), clean(hyp)
    if not r: return 1.0
    return jiwer.wer(' '.join(r), ' '.join(h))

def calc_tone_acc(ref, hyp):
    r, h = clean(ref), clean(hyp)
    if not r: return 0.0
    rp = [p[0] for p in pinyin(r, style=Style.TONE3)]
    hp = [p[0] for p in pinyin(h, style=Style.TONE3)]
    correct_char, correct_tone = 0, 0
    for i in range(min(len(r), len(h), len(rp), len(hp))):
        if r[i] == h[i]:
            correct_char += 1
            if rp[i] == hp[i]:
                correct_tone += 1
    return (correct_tone / correct_char if correct_char > 0 else 0.0)

# 加载测试数据
print("\n加载测试数据...")
test_files = sorted(glob.glob("/data/AD_predict/data/raw/seniortalk_full/sentence_data/test-*.parquet"))
df = pd.read_parquet(test_files[0])
df = df.sample(min(NUM_SAMPLES, len(df)), random_state=42).reset_index(drop=True)
print(f"测试样本数: {len(df)}")

# 加载 FunASR
print("\n加载 FunASR SenseVoice 模型...")
try:
    from funasr import AutoModel
    print("  ✓ FunASR 已安装")
except ImportError:
    print("  Installing FunASR...")
    import subprocess
    subprocess.run(["pip", "install", "-U", "funasr", "funasr-onnx", "modelscope"], check=True)
    from funasr import AutoModel

try:
    # 使用 SenseVoice 多语言模型
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        device="cuda:0",
        disable_update=True,
        hub="ms"  # 使用 modelscope hub
    )
    print("  ✓ FunASR SenseVoice 加载成功")
except Exception as e:
    print(f"  ✗ 加载失败: {e}")
    print("\n尝试备用方法...")
    try:
        model = AutoModel(
            model="paraformer-zh",
            device="cuda:0",
            disable_update=True
        )
        print("  ✓ FunASR Paraformer 加载成功（备用）")
    except Exception as e2:
        print(f"  ✗ 备用方法也失败: {e2}")
        exit(1)

# 评估
print(f"\n{'='*80}")
print("开始评估...")
print(f"{'='*80}")

wers, cers, tone_accs = [], [], []
details = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
    try:
        # 获取音频
        audio_bytes = bytes(row['audio'])
        import soundfile as sf
        import io
        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
        
        # FunASR 推理
        result = model.generate(
            input=audio_array,
            batch_size_s=300
        )
        
        # 解析结果
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                hyp = result[0].get('text', '')
            else:
                hyp = str(result[0])
        else:
            hyp = str(result)
        
        ref = row['text']
        
        # 计算指标
        cer = calc_cer(ref, hyp)
        wer = calc_wer(ref, hyp)
        tone = calc_tone_acc(ref, hyp)
        
        wers.append(wer)
        cers.append(cer)
        tone_accs.append(tone)
        
        details.append({
            "sample_id": idx,
            "reference": ref,
            "hypothesis": hyp,
            "cer": float(cer),
            "wer": float(wer),
            "tone_accuracy": float(tone)
        })
        
    except Exception as e:
        print(f"\n错误 {idx}: {e}")
        continue

# 汇总结果
print(f"\n{'='*80}")
print("评估结果")
print(f"{'='*80}")

result = {
    "model": "FunASR SenseVoice",
    "samples": len(wers),
    "avg_wer": float(np.mean(wers)) if wers else None,
    "avg_cer": float(np.mean(cers)) if cers else None,
    "avg_tone_acc": float(np.mean(tone_accs)) if tone_accs else None,
    "std_wer": float(np.std(wers)) if wers else None,
    "std_cer": float(np.std(cers)) if cers else None,
    "timestamp": datetime.now().isoformat()
}

print(f"\n模型: {result['model']}")
print(f"样本数: {result['samples']}")
print(f"平均 WER: {result['avg_wer']:.4f} (±{result['std_wer']:.4f})")
print(f"平均 CER: {result['avg_cer']:.4f} (±{result['std_cer']:.4f})")
print(f"平均音调准确率: {result['avg_tone_acc']:.4f}")

# 保存结果
output_file = f"{OUTPUT_DIR}/funasr_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        "summary": result,
        "details": details
    }, f, indent=2, ensure_ascii=False)

print(f"\n结果已保存到: {output_file}")
print(f"\n{'='*80}")
print("✓ FunASR 评估完成!")
print(f"{'='*80}")


