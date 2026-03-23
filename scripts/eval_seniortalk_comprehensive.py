#!/usr/bin/env python3
"""
SeniorTalk 测试集全面评估脚本
评估指标：
1. WER (词错误率)
2. CER (字错误率)  
3. 字准确率
4. 音调准确率（字正确时才计算）
"""

import os
# 在导入任何 Hugging Face 相关库之前，设置缓存到用户目录
HOME_DIR = os.path.expanduser("~")
HF_CACHE_DIR = os.path.join(HOME_DIR, ".cache", "huggingface")
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("HF_DATASETS_CACHE", HF_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
import jiwer
from pypinyin import pinyin, Style
import re
import os
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime

# ==============================================================================
# 1. 配置区域
# ==============================================================================

# 模型配置 - 使用本地可用的模型
MODEL_CONFIGS = {
    "whisper_medium_baseline": {
        "base_model": "openai/whisper-medium",
        "adapter_path": None,  # 原始模型
        "name": "Whisper-Medium (原始)",
        "type": "baseline"
    },
    "best_model": {
        "base_model": "openai/whisper-medium",
        "adapter_path": "/home/saisai/AD_predict/AD_predict/models/best_model",
        "name": "Best Model (最优模型)",
        "type": "finetuned"
    },
    "dialect_final": {
        "base_model": "openai/whisper-medium",
        "adapter_path": "/home/saisai/AD_predict/AD_predict/whisper_lora_dialect/final_adapter",
        "name": "Dialect Final Adapter",
        "type": "finetuned"
    },
    "dialect_ckpt60": {
        "base_model": "openai/whisper-medium",
        "adapter_path": "/home/saisai/AD_predict/AD_predict/whisper_lora_dialect/checkpoint-60",
        "name": "Dialect Checkpoint-60",
        "type": "finetuned"
    }
}

# 数据配置（允许通过环境变量覆盖）
# 设置 USE_ONLINE_DATASET=1 可直接从 HF 在线加载
USE_ONLINE_DATASET = os.environ.get("USE_ONLINE_DATASET", "0") == "1"
SENIORTALK_TEST_PATH = os.environ.get(
    "SENIORTALK_TEST_PATH",
    "/data/AD_predict/data/raw/seniortalk_full/sentence_data/test",
)

# 输出目录放在用户目录，避免 /data 写权限问题
OUTPUT_DIR = os.path.join(HOME_DIR, "AD_predict_results", "seniortalk_evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 评估配置
NUM_TEST_SAMPLES = 100  # 测试样本数量，可根据数据集大小调整
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1

print(f"Using device: {DEVICE}")
print(f"Output directory: {OUTPUT_DIR}")

# ==============================================================================
# 2. 文本处理和评估指标函数
# ==============================================================================

def clean_text(text):
    """清洗文本，去除标点和空格"""
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    return text.strip()

def get_pinyin_with_tone(text):
    """获取带声调的拼音"""
    return [p[0] for p in pinyin(text, style=Style.TONE3, errors='ignore')]

def get_pinyin_no_tone(text):
    """获取不带声调的拼音"""
    return [p[0] for p in pinyin(text, style=Style.NORMAL, errors='ignore')]

def compute_wer(reference, hypothesis):
    """计算词错误率 (Word Error Rate)"""
    ref_clean = clean_text(reference)
    hyp_clean = clean_text(hypothesis)
    
    if not ref_clean:
        return 1.0
    
    # 对于中文，将每个字作为一个词
    ref_words = ' '.join(list(ref_clean))
    hyp_words = ' '.join(list(hyp_clean))
    
    return jiwer.wer(ref_words, hyp_words)

def compute_cer(reference, hypothesis):
    """计算字错误率 (Character Error Rate)"""
    ref_clean = clean_text(reference)
    hyp_clean = clean_text(hypothesis)
    
    if not ref_clean:
        return 1.0
    
    return jiwer.cer(ref_clean, hyp_clean)

def compute_character_accuracy(reference, hypothesis):
    """计算字准确率"""
    ref_clean = clean_text(reference)
    hyp_clean = clean_text(hypothesis)
    
    if not ref_clean:
        return 0.0
    
    # 计算正确的字符数
    correct = sum(1 for r, h in zip(ref_clean, hyp_clean) if r == h)
    
    return correct / len(ref_clean)

def compute_tone_accuracy(reference, hypothesis):
    """
    计算音调准确率
    只在字正确的情况下计算音调准确率
    """
    ref_clean = clean_text(reference)
    hyp_clean = clean_text(hypothesis)
    
    if not ref_clean:
        return 0.0, 0
    
    # 获取拼音（带声调）
    ref_pinyin = get_pinyin_with_tone(ref_clean)
    hyp_pinyin = get_pinyin_with_tone(hyp_clean)
    
    # 获取拼音（不带声调）
    ref_pinyin_no_tone = get_pinyin_no_tone(ref_clean)
    hyp_pinyin_no_tone = get_pinyin_no_tone(hyp_clean)
    
    # 只统计字正确的情况
    correct_tone_count = 0
    correct_char_count = 0
    
    min_len = min(len(ref_clean), len(hyp_clean), len(ref_pinyin), len(hyp_pinyin))
    
    for i in range(min_len):
        # 字符相同
        if ref_clean[i] == hyp_clean[i]:
            correct_char_count += 1
            # 检查声调是否也相同
            if ref_pinyin[i] == hyp_pinyin[i]:
                correct_tone_count += 1
    
    if correct_char_count == 0:
        return 0.0, 0
    
    tone_accuracy = correct_tone_count / correct_char_count
    return tone_accuracy, correct_char_count

# ==============================================================================
# 3. 模型加载函数
# ==============================================================================

def load_model(config):
    """加载模型"""
    print(f"\nLoading model: {config['name']}")
    print(f"  Base model: {config['base_model']}")
    
    try:
        # 加载processor
        processor = WhisperProcessor.from_pretrained(config['base_model'])
        
        # 加载基础模型
        model = WhisperForConditionalGeneration.from_pretrained(
            config['base_model'],
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        
        # 如果有adapter，加载adapter
        if config['adapter_path']:
            print(f"  Loading adapter from: {config['adapter_path']}")
            if os.path.exists(config['adapter_path']):
                model = PeftModel.from_pretrained(model, config['adapter_path'])
                model = model.merge_and_unload()  # 合并adapter权重
                print("  ✓ Adapter loaded and merged")
            else:
                print(f"  ✗ Adapter path not found, using base model only")
        
        model.to(DEVICE)
        model.eval()
        
        print(f"  ✓ Model loaded successfully")
        return processor, model
        
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return None, None

# ==============================================================================
# 4. 评估函数
# ==============================================================================

def evaluate_model(model_key, config, test_dataset, num_samples=None):
    """评估单个模型"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {config['name']}")
    print(f"{'='*80}")
    
    # 加载模型
    processor, model = load_model(config)
    
    if model is None or processor is None:
        return None
    
    # 准备测试数据
    if num_samples and num_samples < len(test_dataset):
        # 均匀采样
        indices = np.linspace(0, len(test_dataset)-1, num_samples, dtype=int)
        test_samples = [test_dataset[int(i)] for i in indices]
    else:
        test_samples = test_dataset
    
    print(f"Testing on {len(test_samples)} samples...")
    
    # 评估指标累积
    results = {
        "wer_scores": [],
        "cer_scores": [],
        "char_accuracy_scores": [],
        "tone_accuracy_scores": [],
        "predictions": []
    }
    
    # 逐样本推理
    for idx, sample in enumerate(tqdm(test_samples, desc="Evaluating")):
        try:
            # 获取音频和参考文本
            audio_array = sample['audio']['array']
            sampling_rate = sample['audio']['sampling_rate']
            reference_text = sample['text']
            
            # 预处理音频
            inputs = processor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            )
            
            input_features = inputs.input_features.to(DEVICE)
            
            # 生成
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    language="zh",
                    task="transcribe",
                    max_length=225
                )
            
            # 解码
            hypothesis_text = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            # 计算各项指标
            wer = compute_wer(reference_text, hypothesis_text)
            cer = compute_cer(reference_text, hypothesis_text)
            char_acc = compute_character_accuracy(reference_text, hypothesis_text)
            tone_acc, correct_chars = compute_tone_accuracy(reference_text, hypothesis_text)
            
            results["wer_scores"].append(wer)
            results["cer_scores"].append(cer)
            results["char_accuracy_scores"].append(char_acc)
            results["tone_accuracy_scores"].append(tone_acc)
            
            results["predictions"].append({
                "sample_id": idx,
                "reference": reference_text,
                "hypothesis": hypothesis_text,
                "wer": float(wer),
                "cer": float(cer),
                "char_accuracy": float(char_acc),
                "tone_accuracy": float(tone_acc),
                "correct_chars_for_tone": correct_chars
            })
            
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            continue
    
    # 计算平均指标
    summary = {
        "model_key": model_key,
        "model_name": config['name'],
        "model_type": config['type'],
        "num_samples": len(test_samples),
        "num_evaluated": len(results["wer_scores"]),
        "avg_wer": float(np.mean(results["wer_scores"])) if results["wer_scores"] else None,
        "avg_cer": float(np.mean(results["cer_scores"])) if results["cer_scores"] else None,
        "avg_char_accuracy": float(np.mean(results["char_accuracy_scores"])) if results["char_accuracy_scores"] else None,
        "avg_tone_accuracy": float(np.mean(results["tone_accuracy_scores"])) if results["tone_accuracy_scores"] else None,
        "std_wer": float(np.std(results["wer_scores"])) if results["wer_scores"] else None,
        "std_cer": float(np.std(results["cer_scores"])) if results["cer_scores"] else None,
    }
    
    print(f"\nResults Summary:")
    print(f"  Average WER: {summary['avg_wer']:.4f} (±{summary['std_wer']:.4f})")
    print(f"  Average CER: {summary['avg_cer']:.4f} (±{summary['std_cer']:.4f})")
    print(f"  Average Char Accuracy: {summary['avg_char_accuracy']:.4f}")
    print(f"  Average Tone Accuracy: {summary['avg_tone_accuracy']:.4f}")
    
    # 清理GPU内存
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "summary": summary,
        "detailed_results": results["predictions"]
    }

# ==============================================================================
# 5. 主函数
# ==============================================================================

def main():
    print("="*80)
    print("SeniorTalk Test Set Comprehensive Evaluation")
    print("="*80)
    
    # 加载测试数据
    print(f"\nLoading test dataset...")
    try:
        if USE_ONLINE_DATASET:
            print("  Mode: Online loading from Hugging Face")
            print("  Dataset: evan0617/seniortalk")
            # 设置镜像站（如果在国内）
            hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
            if "hf-mirror.com" in hf_endpoint or "modelscope" in hf_endpoint:
                print(f"  Using mirror: {hf_endpoint}")
                os.environ["HF_ENDPOINT"] = hf_endpoint
            
            test_dataset = load_dataset(
                "evan0617/seniortalk",
                "sentence_data",
                split="test",
                trust_remote_code=True
            )
            print(f"✓ Loaded {len(test_dataset)} test samples from Hugging Face")
        else:
            print(f"  Mode: Local dataset from: {SENIORTALK_TEST_PATH}")
            test_dataset = load_from_disk(SENIORTALK_TEST_PATH)
            print(f"✓ Loaded {len(test_dataset)} test samples from local")
        
        # 显示数据集信息
        print(f"\nDataset info:")
        if len(test_dataset) > 0:
            print(f"  Sample keys: {list(test_dataset[0].keys())}")
            print(f"  First sample text: {test_dataset[0]['text'][:100]}...")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 检查可用模型
    print(f"\n{'='*80}")
    print("Checking available models...")
    print(f"{'='*80}")
    
    available_models = {}
    for key, config in MODEL_CONFIGS.items():
        if config['adapter_path'] is None:
            available_models[key] = config
            print(f"✓ {config['name']}: Baseline model (no adapter needed)")
        elif os.path.exists(config['adapter_path']):
            available_models[key] = config
            print(f"✓ {config['name']}: {config['adapter_path']}")
        else:
            print(f"✗ {config['name']}: NOT FOUND at {config['adapter_path']}")
    
    if not available_models:
        print("\n✗ No models available for evaluation!")
        return
    
    print(f"\nTotal available models: {len(available_models)}")
    
    # 评估所有模型
    all_results = {}
    
    for model_key in available_models:
        config = available_models[model_key]
        result = evaluate_model(model_key, config, test_dataset, NUM_TEST_SAMPLES)
        
        if result:
            all_results[model_key] = result
            
            # 保存单个模型的详细结果
            detail_file = Path(OUTPUT_DIR) / f"{model_key}_detailed_results.json"
            with open(detail_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  Detailed results saved to: {detail_file}")
    
    # 生成汇总报告
    print(f"\n{'='*80}")
    print("Generating Summary Report")
    print(f"{'='*80}")
    
    summary_data = []
    for model_key, result in all_results.items():
        summary_data.append(result['summary'])
    
    # 保存汇总JSON
    summary_file = Path(OUTPUT_DIR) / "evaluation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "test_dataset": SENIORTALK_TEST_PATH,
            "num_samples_per_model": NUM_TEST_SAMPLES,
            "device": DEVICE,
            "models_evaluated": summary_data
        }, f, indent=2, ensure_ascii=False)
    print(f"✓ Summary saved to: {summary_file}")
    
    # 生成对比表格
    df = pd.DataFrame(summary_data)
    df = df.sort_values('avg_cer')  # 按CER排序
    
    # 保存CSV
    csv_file = Path(OUTPUT_DIR) / "evaluation_comparison.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"✓ Comparison table saved to: {csv_file}")
    
    # 打印对比表格
    print(f"\n{'='*80}")
    print("Model Performance Comparison (Sorted by CER)")
    print(f"{'='*80}")
    print(df[['model_name', 'avg_wer', 'avg_cer', 'avg_char_accuracy', 'avg_tone_accuracy']].to_string(index=False))
    
    print(f"\n{'='*80}")
    print("✓ Evaluation completed successfully!")
    print(f"{'='*80}")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()


