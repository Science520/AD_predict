#!/usr/bin/env python3
"""
使用SeniorTalk数据评估Whisper模型
策略：用Whisper-large-v3生成高质量伪标签作为参考，评估微调模型
"""
import os
import sys
import json
import tarfile
from pathlib import Path
from tqdm import tqdm
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import pandas as pd
from pypinyin import lazy_pinyin
from jiwer import wer

# 配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TAR_PATH = "/data/AD_predict/data/raw/audio/seniortalk_asr_single/sentence_data/wav/train/train-0001.tar"
TEMP_DIR = "/data/AD_predict/data/seniortalk_eval"  # 使用/data目录，避免占用系统盘空间
NUM_SAMPLES = 50  # 评估样本数

# 模型配置
MODEL_CONFIGS = [
    {
        "name": "原始Whisper-Medium",
        "id": "whisper_medium_base",
        "base_model": "openai/whisper-medium",
        "lora_path": None,
    },
    {
        "name": "Exp1: 高Rank (step 100, 最佳)",
        "id": "exp1_high_rank_100",
        "base_model": "openai/whisper-medium",
        "lora_path": "/data/AD_predict/all_experiments_20251022_140017/exp1_high_rank/checkpoint-100",
    },
    {
        "name": "Exp2: 低学习率 (step 1100)",
        "id": "exp2_low_lr_1100",
        "base_model": "openai/whisper-medium",
        "lora_path": "/data/AD_predict/all_experiments_20251022_140017/exp2_low_lr/checkpoint-1100",
    },
    {
        "name": "Exp3: 大Batch (step 750, 最佳)",
        "id": "exp3_large_batch_750",
        "base_model": "openai/whisper-medium",
        "lora_path": "/data/AD_predict/all_experiments_20251022_140017/exp3_large_batch/checkpoint-750",
    },
    {
        "name": "Exp4: 激进学习率 (step 500)",
        "id": "exp4_aggressive_500",
        "base_model": "openai/whisper-medium",
        "lora_path": "/data/AD_predict/all_experiments_20251022_140017/exp4_aggressive/checkpoint-500",
    },
]

# ============================================================================
# 数据准备
# ============================================================================

def extract_samples_from_tar(tar_path: str, output_dir: str, num_samples: int = 50):
    """从tar包中提取指定数量的音频样本"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 从tar包提取 {num_samples} 个样本...")
    print(f"   tar包: {tar_path}")
    print(f"   输出到: {output_dir}")
    
    extracted_files = []
    
    with tarfile.open(tar_path, 'r') as tar:
        # 获取所有wav文件
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith('.wav')]
        
        # 随机采样（或取前N个）
        import random
        random.seed(42)
        if len(members) > num_samples:
            members = random.sample(members, num_samples)
        
        print(f"   提取 {len(members)} 个文件...")
        
        for member in tqdm(members, desc="解压"):
            tar.extract(member, output_path)
            extracted_path = output_path / member.name
            
            # 检查音频是否有效
            try:
                y, sr = librosa.load(str(extracted_path), sr=16000, duration=1)
                if len(y) > 0:
                    extracted_files.append(str(extracted_path))
                else:
                    print(f"⚠️ 跳过空音频: {member.name}")
            except Exception as e:
                print(f"⚠️ 跳过损坏音频: {member.name} ({e})")
    
    print(f"✅ 成功提取 {len(extracted_files)} 个有效样本")
    return extracted_files


def generate_reference_transcripts(audio_files: list, output_json: str):
    """使用Whisper-large-v3生成高质量参考转录"""
    
    print(f"\n🤖 使用 Whisper-large-v3 生成参考转录...")
    print(f"   设备: {DEVICE}")
    
    # 加载大模型
    model_name = "openai/whisper-large-v3"
    print(f"   加载模型: {model_name}")
    
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✅ 模型加载完成")
    
    # 生成转录
    reference_data = []
    
    for audio_path in tqdm(audio_files, desc="生成参考转录"):
        try:
            # 加载音频
            audio_array, sr = librosa.load(audio_path, sr=16000)
            
            # 处理
            input_features = processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(DEVICE)
            
            # 生成
            with torch.no_grad():
                forced_decoder_ids = processor.get_decoder_prompt_ids(
                    language="zh",
                    task="transcribe"
                )
                predicted_ids = model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_length=448,
                    num_beams=5
                )
            
            transcription = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0].strip()
            
            reference_data.append({
                "audio_path": audio_path,
                "reference_text": transcription,
                "duration": len(audio_array) / sr
            })
            
        except Exception as e:
            print(f"⚠️ 生成失败 {audio_path}: {e}")
            continue
    
    # 保存
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(reference_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 参考转录生成完成: {len(reference_data)} 条")
    print(f"   保存到: {output_json}")
    
    return reference_data


# ============================================================================
# 评估函数
# ============================================================================

def clean_text(text: str) -> str:
    """清理文本"""
    import re
    # 移除标点和空格
    text = re.sub(r'[，。！？、；：""''（）《》【】\s]+', '', text)
    return text.lower()


def get_pinyin_list(text: str) -> list:
    """获取拼音列表"""
    clean = clean_text(text)
    if not clean:
        return []
    return lazy_pinyin(clean)


def compute_cer(reference: str, hypothesis: str) -> float:
    """计算字错误率 (CER)"""
    ref_clean = clean_text(reference)
    hyp_clean = clean_text(hypothesis)
    
    if not ref_clean:
        return 1.0 if hyp_clean else 0.0
    
    try:
        # 使用WER计算，但基于字符
        error_rate = wer(ref_clean, hyp_clean)
        return min(error_rate, 1.0)
    except:
        return 1.0


def compute_pinyin_accuracy(reference: str, hypothesis: str) -> float:
    """计算拼音准确率"""
    ref_pinyin = get_pinyin_list(reference)
    hyp_pinyin = get_pinyin_list(hypothesis)
    
    if not ref_pinyin:
        return 1.0 if not hyp_pinyin else 0.0
    
    try:
        # 连接成字符串计算WER
        ref_str = ' '.join(ref_pinyin)
        hyp_str = ' '.join(hyp_pinyin)
        error_rate = wer(ref_str, hyp_str)
        return max(0.0, 1.0 - error_rate)
    except:
        return 0.0


def load_model(base_model: str, lora_path: str = None):
    """加载模型（基础模型或LoRA微调）"""
    
    processor = WhisperProcessor.from_pretrained(base_model)
    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    
    if lora_path and os.path.exists(lora_path):
        print(f"   加载LoRA: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    
    model = model.to(DEVICE)
    model.eval()
    
    return processor, model


def evaluate_model(model_config: dict, test_data: list):
    """评估单个模型"""
    
    print(f"\n{'='*80}")
    print(f"评估模型: {model_config['name']}")
    print(f"{'='*80}")
    
    # 加载模型
    processor, model = load_model(
        model_config['base_model'],
        model_config.get('lora_path')
    )
    
    results = []
    
    for item in tqdm(test_data, desc="评估"):
        audio_path = item['audio_path']
        reference_text = item['reference_text']
        
        try:
            # 加载音频
            audio_array, sr = librosa.load(audio_path, sr=16000)
            
            # 处理
            input_features = processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(DEVICE)
            
            # 生成
            with torch.no_grad():
                forced_decoder_ids = processor.get_decoder_prompt_ids(
                    language="zh",
                    task="transcribe"
                )
                predicted_ids = model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_length=448
                )
            
            predicted_text = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0].strip()
            
            # 计算指标
            cer = compute_cer(reference_text, predicted_text)
            char_accuracy = max(0.0, (1 - cer) * 100)
            pinyin_acc = compute_pinyin_accuracy(reference_text, predicted_text) * 100
            
            results.append({
                "audio_file": Path(audio_path).name,
                "reference": reference_text,
                "predicted": predicted_text,
                "char_accuracy": char_accuracy,
                "pinyin_accuracy": pinyin_acc
            })
            
        except Exception as e:
            print(f"⚠️ 评估失败 {audio_path}: {e}")
            continue
    
    # 计算平均指标
    if results:
        avg_char_acc = sum(r['char_accuracy'] for r in results) / len(results)
        avg_pinyin_acc = sum(r['pinyin_accuracy'] for r in results) / len(results)
    else:
        avg_char_acc = 0.0
        avg_pinyin_acc = 0.0
    
    print(f"\n📊 评估完成:")
    print(f"   样本数: {len(results)}")
    print(f"   平均字正确率: {avg_char_acc:.2f}%")
    print(f"   平均拼音准确率: {avg_pinyin_acc:.2f}%")
    
    # 释放内存
    del model
    del processor
    torch.cuda.empty_cache()
    
    return {
        "model_name": model_config['name'],
        "model_id": model_config['id'],
        "avg_char_accuracy": avg_char_acc,
        "avg_pinyin_accuracy": avg_pinyin_acc,
        "num_samples": len(results),
        "details": results
    }


# ============================================================================
# 主流程
# ============================================================================

def main():
    print("="*80)
    print("🎯 SeniorTalk数据集评估")
    print("="*80)
    
    # 步骤1: 提取音频样本
    print(f"\n步骤1: 提取 {NUM_SAMPLES} 个音频样本")
    print("-"*80)
    audio_files = extract_samples_from_tar(TAR_PATH, TEMP_DIR, NUM_SAMPLES)
    
    if not audio_files:
        print("❌ 没有提取到有效音频，退出")
        sys.exit(1)
    
    # 步骤2: 生成参考转录
    print(f"\n步骤2: 生成参考转录（使用Whisper-large-v3）")
    print("-"*80)
    reference_json = os.path.join(TEMP_DIR, "reference_transcripts.json")
    
    if os.path.exists(reference_json):
        print(f"⏭️ 发现已有参考转录: {reference_json}")
        with open(reference_json, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    else:
        test_data = generate_reference_transcripts(audio_files, reference_json)
    
    if not test_data:
        print("❌ 没有生成参考转录，退出")
        sys.exit(1)
    
    print(f"\n✅ 准备就绪，共 {len(test_data)} 个测试样本")
    
    # 显示一些参考转录示例
    print(f"\n📝 参考转录示例（前3条）:")
    for i, item in enumerate(test_data[:3]):
        print(f"\n{i+1}. {Path(item['audio_path']).name}")
        print(f"   转录: {item['reference_text']}")
    
    # 步骤3: 评估所有模型
    print(f"\n步骤3: 评估所有模型")
    print("-"*80)
    
    all_results = []
    
    for model_config in MODEL_CONFIGS:
        result = evaluate_model(model_config, test_data)
        all_results.append(result)
    
    # 步骤4: 生成报告
    print(f"\n步骤4: 生成评估报告")
    print("-"*80)
    
    # 保存详细结果
    output_dir = "/data/AD_predict/seniortalk_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    detailed_json = os.path.join(output_dir, "detailed_results.json")
    with open(detailed_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # 生成CSV汇总
    summary_data = []
    for result in all_results:
        summary_data.append({
            "模型": result['model_name'],
            "模型ID": result['model_id'],
            "字正确率(%)": f"{result['avg_char_accuracy']:.2f}",
            "拼音准确率(%)": f"{result['avg_pinyin_accuracy']:.2f}",
            "样本数": result['num_samples']
        })
    
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, "summary.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 生成Markdown报告
    md_report = f"""# 🎯 SeniorTalk评估报告

**评估数据**: SeniorTalk数据集（{len(test_data)}个样本）
**参考转录**: Whisper-large-v3生成（高质量伪标签）
**评估指标**: 字正确率 (1-CER) + 拼音准确率

---

## 📊 结果汇总

{df.to_markdown(index=False)}

---

## 💡 关键发现

"""
    
    # 找最佳模型
    best_model = max(all_results, key=lambda x: x['avg_char_accuracy'])
    worst_model = min(all_results, key=lambda x: x['avg_char_accuracy'])
    
    md_report += f"""
1. **最佳模型**: {best_model['model_name']}
   - 字正确率: {best_model['avg_char_accuracy']:.2f}%
   - 拼音准确率: {best_model['avg_pinyin_accuracy']:.2f}%

2. **基线模型** (原始Whisper-Medium):
   - 字正确率: {[r for r in all_results if r['model_id'] == 'whisper_medium_base'][0]['avg_char_accuracy']:.2f}%
   
3. **性能提升**:
   - 最佳模型相比基线: +{best_model['avg_char_accuracy'] - [r for r in all_results if r['model_id'] == 'whisper_medium_base'][0]['avg_char_accuracy']:.2f}%

---

## 📝 转录示例对比（前5条）

"""
    
    # 添加转录对比示例
    for i in range(min(5, len(test_data))):
        item = test_data[i]
        md_report += f"\n### 样本 {i+1}: {Path(item['audio_path']).name}\n\n"
        md_report += f"**参考转录** (Whisper-large-v3):\n```\n{item['reference_text']}\n```\n\n"
        
        for result in all_results:
            detail = result['details'][i] if i < len(result['details']) else None
            if detail:
                md_report += f"**{result['model_name']}** (字正确率: {detail['char_accuracy']:.1f}%):\n```\n{detail['predicted']}\n```\n\n"
        
        md_report += "---\n\n"
    
    md_path = os.path.join(output_dir, "SENIORTALK_EVALUATION_REPORT.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    
    # 打印汇总
    print("\n" + "="*80)
    print("✅ 评估完成！")
    print("="*80)
    print(f"\n📊 结果汇总:\n")
    print(df.to_string(index=False))
    print(f"\n💾 详细结果已保存:")
    print(f"   - 详细JSON: {detailed_json}")
    print(f"   - CSV汇总: {csv_path}")
    print(f"   - Markdown报告: {md_path}")
    print("="*80)


if __name__ == "__main__":
    main()

