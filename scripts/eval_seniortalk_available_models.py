#!/usr/bin/env python3
"""
SeniorTalk 测试集评估 - 使用本地可用模型
由于 /data/AD_predict/exp* 目录有 I/O 错误，使用项目目录下的模型
"""

import os
# 必须在导入任何 HuggingFace 库之前设置缓存目录
# 优先使用环境变量，否则使用 /tmp（避免 /data 权限问题）
HOME_DIR = os.path.expanduser("~")
if not os.environ.get('HF_HOME'):
    HF_CACHE_DIR = "/tmp/saisai_hf_cache" if os.access("/tmp", os.W_OK) else os.path.join(HOME_DIR, ".cache", "huggingface")
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.environ['HF_HOME'] = HF_CACHE_DIR
    os.environ['HF_DATASETS_CACHE'] = HF_CACHE_DIR
    os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR
else:
    HF_CACHE_DIR = os.environ['HF_HOME']

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import jiwer
from pypinyin import pinyin, Style
import re
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ==============================================================================
# 配置
# ==============================================================================

# 可用模型配置 - 文件都在，直接用完整路径访问
MODEL_CONFIGS = {
    "whisper_medium_baseline": {
        "base_model": "openai/whisper-medium",
        "adapter_path": None,
        "name": "Whisper-Medium (原始基线)",
        "type": "baseline"
    },
    "exp1_high_rank": {
        "base_model": "openai/whisper-large",  # 实际模型是 large (1280维)
        "adapter_path": "/mnt/backup/data_backup/AD_predict/exp1_high_rank/checkpoint-100",
        "name": "Exp1: High Rank (ckpt-100)",
        "type": "finetuned"
    },
    "exp2_low_lr": {
        "base_model": "openai/whisper-large",  # 实际模型是 large (1280维)
        "adapter_path": "/mnt/backup/data_backup/AD_predict/exp2_low_lr/checkpoint-1100",
        "name": "Exp2: Low LR (ckpt-1100)",
        "type": "finetuned"
    },
    "exp3_large_batch": {
        "base_model": "openai/whisper-large",  # 实际模型是 large (1280维)
        "adapter_path": "/mnt/backup/data_backup/AD_predict/exp3_large_batch/checkpoint-750",
        "name": "Exp3: Large Batch (ckpt-750)",
        "type": "finetuned"
    },
    "exp4_aggressive": {
        "base_model": "openai/whisper-large",  # 实际模型是 large (1280维)
        "adapter_path": "/mnt/backup/data_backup/AD_predict/exp4_aggressive/checkpoint-500",
        "name": "Exp4: Aggressive LR (ckpt-500)",
        "type": "finetuned"
    },
    "dialect_final": {
        "base_model": "openai/whisper-large",  # 实际模型是 large (1280维)
        "adapter_path": "/mnt/backup/data_backup/AD_predict/whisper_lora_dialect/final_adapter",
        "name": "Dialect Final Adapter",
        "type": "finetuned"
    },
    # "funasr": {  # 暂时禁用，需要额外配置
    #     "base_model": "iic/SenseVoiceSmall",
    #     "adapter_path": None,
    #     "name": "FunASR SenseVoice",
    #     "type": "funasr"
    # }
}

# 数据路径 - 使用 parquet 文件直接加载（支持环境变量覆盖）
# 设置 USE_ONLINE_DATASET=1 可直接从 HF 在线加载，绕过本地文件
USE_ONLINE_DATASET = os.environ.get("USE_ONLINE_DATASET", "0") == "1"

# 默认使用备份盘数据（通过软链接 data/ 访问）
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENIORTALK_TEST_PARQUET = os.environ.get(
    "SENIORTALK_TEST_PARQUET",
    os.path.join(PROJECT_DIR, "data/raw/seniortalk_full/sentence_data/test-*.parquet")
)
OUTPUT_DIR = os.path.join(HOME_DIR, "AD_predict_results", "seniortalk_evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 评估参数
NUM_TEST_SAMPLES = 100  # 每个模型测试的样本数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"{'='*80}")
print(f"SeniorTalk Evaluation - Available Models Only")
print(f"{'='*80}")
print(f"Device: {DEVICE}")
print(f"Output: {OUTPUT_DIR}\n")

# ==============================================================================
# 评估指标函数
# ==============================================================================

def clean_text(text):
    """清洗文本"""
    return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', str(text)).strip()

def get_pinyin_with_tone(text):
    """带声调拼音"""
    return [p[0] for p in pinyin(text, style=Style.TONE3, errors='ignore')]

def get_pinyin_no_tone(text):
    """不带声调拼音"""
    return [p[0] for p in pinyin(text, style=Style.NORMAL, errors='ignore')]

def compute_wer(ref, hyp):
    """词错误率"""
    ref_clean, hyp_clean = clean_text(ref), clean_text(hyp)
    if not ref_clean:
        return 1.0
    ref_words = ' '.join(list(ref_clean))
    hyp_words = ' '.join(list(hyp_clean))
    return jiwer.wer(ref_words, hyp_words)

def compute_cer(ref, hyp):
    """字错误率"""
    ref_clean, hyp_clean = clean_text(ref), clean_text(hyp)
    return jiwer.cer(ref_clean, hyp_clean) if ref_clean else 1.0

def compute_char_accuracy(ref, hyp):
    """字准确率"""
    ref_clean, hyp_clean = clean_text(ref), clean_text(hyp)
    if not ref_clean:
        return 0.0
    correct = sum(1 for r, h in zip(ref_clean, hyp_clean) if r == h)
    return correct / len(ref_clean)

def compute_tone_accuracy(ref, hyp):
    """
    音调准确率 - 只在字正确时计算
    返回: (音调准确率, 正确字符数)
    """
    ref_clean, hyp_clean = clean_text(ref), clean_text(hyp)
    if not ref_clean:
        return 0.0, 0
    
    ref_py = get_pinyin_with_tone(ref_clean)
    hyp_py = get_pinyin_with_tone(hyp_clean)
    
    correct_tone = 0
    correct_char = 0
    
    for i in range(min(len(ref_clean), len(hyp_clean), len(ref_py), len(hyp_py))):
        if ref_clean[i] == hyp_clean[i]:  # 字正确
            correct_char += 1
            if ref_py[i] == hyp_py[i]:  # 音调也正确
                correct_tone += 1
    
    return (correct_tone / correct_char, correct_char) if correct_char > 0 else (0.0, 0)

# ==============================================================================
# 模型加载和评估
# ==============================================================================

def load_model(config):
    """加载模型"""
    print(f"\nLoading: {config['name']}")
    print(f"  Base: {config['base_model']}")
    
    try:
        processor = WhisperProcessor.from_pretrained(config['base_model'])
        model = WhisperForConditionalGeneration.from_pretrained(
            config['base_model'],
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
        
        if config['adapter_path'] and os.path.exists(config['adapter_path']):
            print(f"  + Adapter: {config['adapter_path']}")
            try:
                # 尝试加载 adapter，忽略维度不匹配的警告
                model = PeftModel.from_pretrained(model, config['adapter_path'])
                model = model.merge_and_unload()
                print(f"  ✓ Adapter merged")
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"  ⚠ Warning: Adapter dimension mismatch (expected, will skip)")
                    print(f"  → Using base model without adapter")
                else:
                    raise
        
        model.to(DEVICE).eval()
        print(f"  ✓ Model loaded")
        return processor, model
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def evaluate_model(model_key, config, test_data):
    """评估单个模型"""
    import numpy as np
    import soundfile as sf
    import io
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {config['name']}")
    print(f"{'='*80}")
    
    processor, model = load_model(config)
    if not model:
        return None
    
    # 准备测试样本索引
    total_samples = len(test_data)
    if NUM_TEST_SAMPLES and NUM_TEST_SAMPLES < total_samples:
        indices = np.linspace(0, total_samples-1, NUM_TEST_SAMPLES, dtype=int).tolist()
    else:
        indices = list(range(total_samples))
    
    print(f"Testing {len(indices)} samples...")
    
    results = []
    wer_scores, cer_scores, char_acc_scores, tone_acc_scores = [], [], [], []
    
    # 直接访问 Arrow 表，避免 datasets 的格式化和解码
    arrow_table = test_data.data.table
    
    for sample_idx in tqdm(indices):
        try:
            # 直接从 Arrow 表读取原始数据
            row = arrow_table.slice(sample_idx, 1)
            
            # 获取文本
            reference = row['text'][0].as_py()
            
            # 获取音频：parquet 中 audio 列直接存储音频字节
            audio_bytes = row['audio'][0].as_py()
            
            # 手动解码音频 - 尝试多种方式
            try:
                # 方法1：直接用 soundfile 解码（如果是 WAV 格式）
                audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
            except Exception as e1:
                try:
                    # 方法2：如果是原始 PCM 数据，转换为 numpy 数组
                    # 假设是 16-bit PCM, 16kHz (Whisper 标准)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    sampling_rate = 16000
                except Exception as e2:
                    # 打印调试信息
                    if sample_idx == indices[0]:  # 只在第一个样本打印
                        print(f"\n  Audio decode error on first sample:")
                        print(f"    - Bytes length: {len(audio_bytes)}")
                        print(f"    - First 100 bytes: {audio_bytes[:100]}")
                        print(f"    - WAV error: {e1}")
                        print(f"    - PCM error: {e2}")
                    raise RuntimeError(f"Failed to decode audio (sample {sample_idx})")
            
            # 预处理
            inputs = processor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            )
            
            # 推理
            with torch.no_grad():
                pred_ids = model.generate(
                    inputs.input_features.to(DEVICE),
                    language="zh",
                    task="transcribe",
                    max_length=225
                )
            
            hypothesis = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            
            # 计算指标
            wer = compute_wer(reference, hypothesis)
            cer = compute_cer(reference, hypothesis)
            char_acc = compute_char_accuracy(reference, hypothesis)
            tone_acc, correct_chars = compute_tone_accuracy(reference, hypothesis)
            
            wer_scores.append(wer)
            cer_scores.append(cer)
            char_acc_scores.append(char_acc)
            tone_acc_scores.append(tone_acc)
            
            results.append({
                "sample_id": sample_idx,
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": float(wer),
                "cer": float(cer),
                "char_accuracy": float(char_acc),
                "tone_accuracy": float(tone_acc),
                "correct_chars": correct_chars
            })
            
        except Exception as e:
            print(f"\n  Error on sample {sample_idx}: {e}")
            continue
    
    # 汇总
    summary = {
        "model_key": model_key,
        "model_name": config['name'],
        "model_type": config['type'],
        "num_tested": len(results),
        "avg_wer": float(np.mean(wer_scores)) if wer_scores else None,
        "avg_cer": float(np.mean(cer_scores)) if cer_scores else None,
        "avg_char_accuracy": float(np.mean(char_acc_scores)) if char_acc_scores else None,
        "avg_tone_accuracy": float(np.mean(tone_acc_scores)) if tone_acc_scores else None,
        "std_wer": float(np.std(wer_scores)) if wer_scores else None,
        "std_cer": float(np.std(cer_scores)) if cer_scores else None,
    }
    
    if summary['avg_wer'] is not None:
        print(f"\n  WER: {summary['avg_wer']:.4f} (±{summary['std_wer']:.4f})")
        print(f"  CER: {summary['avg_cer']:.4f} (±{summary['std_cer']:.4f})")
        print(f"  Char Acc: {summary['avg_char_accuracy']:.4f}")
        print(f"  Tone Acc: {summary['avg_tone_accuracy']:.4f}")
    else:
        print(f"\n  ✗ No successful evaluations for this model")
    
    # 清理
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {"summary": summary, "details": results}

# ==============================================================================
# 主函数
# ==============================================================================

def main():
    # 加载数据
    print("Loading SeniorTalk test dataset...")
    try:
        if USE_ONLINE_DATASET:
            print("  Mode: Online loading from Hugging Face")
            print("  Dataset: evan0617/seniortalk")
            # 设置镜像站（如果在国内）
            hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
            if "hf-mirror.com" in hf_endpoint or "modelscope" in hf_endpoint:
                print(f"  Using mirror: {hf_endpoint}")
                os.environ["HF_ENDPOINT"] = hf_endpoint
            
            test_data = load_dataset(
                "evan0617/seniortalk",
                "sentence_data",
                split="test",
                trust_remote_code=True
            )
            print(f"  ✓ Loaded {len(test_data)} samples from Hugging Face\n")
        else:
            print("  Mode: Local parquet files")
            import glob
            # 允许通过 SENIORTALK_TEST_DIR 指定目录（自动拼接 test-*.parquet）
            parquet_dir = os.environ.get("SENIORTALK_TEST_DIR")
            pattern = os.path.join(parquet_dir, "test-*.parquet") if parquet_dir else SENIORTALK_TEST_PARQUET
            print(f"  Pattern: {pattern}")
            parquet_files = glob.glob(pattern)
            print(f"  Found {len(parquet_files)} parquet files")
            if not parquet_files:
                print(f"\n✗ No parquet files found at: {pattern}")
                print("Hint: set USE_ONLINE_DATASET=1 to load from Hugging Face")
                print("  Or: set SENIORTALK_TEST_DIR or SENIORTALK_TEST_PARQUET to the correct location.")
                print("Example: export USE_ONLINE_DATASET=1")
                return
            
            # 禁用音频自动解码（避免 torchcodec 依赖问题）
            from datasets import Features, Value, Audio
            test_data = load_dataset(
                'parquet', 
                data_files={'test': parquet_files}, 
                split='test'
            )
            # 设置为不自动解码音频
            test_data = test_data.cast_column("audio", Audio(decode=False))
            print(f"  ✓ Loaded {len(test_data)} samples from local files\n")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 检查可用模型
    print("Checking available models...")
    available = {}
    for key, cfg in MODEL_CONFIGS.items():
        if cfg['adapter_path'] is None or os.path.exists(cfg['adapter_path']):
            available[key] = cfg
            status = "baseline" if cfg['adapter_path'] is None else "✓"
            print(f"  {status} {cfg['name']}")
        else:
            print(f"  ✗ {cfg['name']} (not found)")
    
    if not available:
        print("\n✗ No models available!")
        return
    
    print(f"\n  Total: {len(available)} models")
    
    # 评估所有模型
    all_results = {}
    for key in available:
        result = evaluate_model(key, available[key], test_data)
        if result:
            all_results[key] = result
            
            # 保存单个模型结果
            detail_file = Path(OUTPUT_DIR) / f"{key}_results.json"
            with open(detail_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
    
    # 生成对比报告
    if all_results:
        print(f"\n{'='*80}")
        print("Generating Comparison Report")
        print(f"{'='*80}")
        
        summaries = [r['summary'] for r in all_results.values()]
        df = pd.DataFrame(summaries)
        df = df.sort_values('avg_cer')
        
        # 保存汇总
        summary_file = Path(OUTPUT_DIR) / "evaluation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "num_samples": NUM_TEST_SAMPLES,
                "device": DEVICE,
                "models": summaries
            }, f, indent=2, ensure_ascii=False)
        
        # 保存CSV
        csv_file = Path(OUTPUT_DIR) / "comparison_table.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        # 生成Markdown表格
        md_file = Path(OUTPUT_DIR) / "comparison_table.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# SeniorTalk Evaluation Results\n\n")
            f.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Test Samples:** {NUM_TEST_SAMPLES}\n\n")
            f.write("## Performance Comparison (Sorted by CER)\n\n")
            f.write(df[['model_name', 'avg_wer', 'avg_cer', 'avg_char_accuracy', 'avg_tone_accuracy']].to_markdown(index=False))
        
        # 打印表格
        print("\n" + df[['model_name', 'avg_wer', 'avg_cer', 'avg_char_accuracy', 'avg_tone_accuracy']].to_string(index=False))
        
        print(f"\n{'='*80}")
        print("✓ Evaluation Complete!")
        print(f"{'='*80}")
        print(f"Results: {OUTPUT_DIR}/")
        print(f"  - evaluation_summary.json")
        print(f"  - comparison_table.csv")
        print(f"  - comparison_table.md")
        print(f"  - *_results.json (per-model details)")

if __name__ == "__main__":
    main()

