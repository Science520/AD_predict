#!/usr/bin/env python3
"""
快速评估 - 使用本地可用模型在 SeniorTalk 测试集上评估
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import pandas as pd
import glob
import os
import json
import numpy as np
from tqdm import tqdm
from pypinyin import pinyin, Style
import re
import jiwer
from datetime import datetime

# 配置
NUM_SAMPLES = 100
OUTPUT_DIR = "/data/AD_predict/experiments/seniortalk_eval_quick"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
    "baseline": {"path": None, "name": "Whisper-Medium原始"},
    "best": {"path": "/home/saisai/AD_predict/AD_predict/models/best_model", "name": "Best Model"},
    "dialect_final": {"path": "/home/saisai/AD_predict/AD_predict/whisper_lora_dialect/final_adapter", "name": "Dialect Final"},
}

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
print("加载测试数据...")
test_files = sorted(glob.glob("/data/AD_predict/data/raw/seniortalk_full/sentence_data/test-*.parquet"))
df = pd.read_parquet(test_files[0])  # 只用第一个文件
df = df.sample(min(NUM_SAMPLES, len(df)), random_state=42).reset_index(drop=True)
print(f"测试样本数: {len(df)}")

# 评估每个模型
results = {}
for key, cfg in MODELS.items():
    if cfg['path'] and not os.path.exists(cfg['path']):
        print(f"跳过 {cfg['name']} - 文件不存在")
        continue
    
    print(f"\n{'='*60}\n评估: {cfg['name']}\n{'='*60}")
    
    # 加载模型
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-medium",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    if cfg['path']:
        model = PeftModel.from_pretrained(model, cfg['path'])
        model = model.merge_and_unload()
    
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    
    # 评估
    wers, cers, tone_accs = [], [], []
    details = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            audio_bytes = bytes(row['audio'])
            import soundfile as sf
            import io
            audio_array, sr = sf.read(io.BytesIO(audio_bytes))
            
            inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt")
            with torch.no_grad():
                pred_ids = model.generate(
                    inputs.input_features.to(model.device),
                    language="zh",
                    max_length=225
                )
            
            hyp = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            ref = row['text']
            
            cer = calc_cer(ref, hyp)
            wer = calc_wer(ref, hyp)
            tone = calc_tone_acc(ref, hyp)
            
            wers.append(wer)
            cers.append(cer)
            tone_accs.append(tone)
            details.append({"ref": ref, "hyp": hyp, "cer": cer})
        except Exception as e:
            print(f"错误 {idx}: {e}")
    
    # 汇总
    result = {
        "model": cfg['name'],
        "samples": len(wers),
        "avg_wer": float(np.mean(wers)),
        "avg_cer": float(np.mean(cers)),
        "avg_tone_acc": float(np.mean(tone_accs)),
    }
    results[key] = result
    print(f"\nWER: {result['avg_wer']:.3f}, CER: {result['avg_cer']:.3f}, Tone: {result['avg_tone_acc']:.3f}")
    
    del model, processor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# 保存结果
with open(f"{OUTPUT_DIR}/results.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}\n总结\n{'='*60}")
for k, v in results.items():
    print(f"{v['model']:20s} CER: {v['avg_cer']:.3f}")

print(f"\n结果已保存到: {OUTPUT_DIR}/results.json")


