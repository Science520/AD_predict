#!/usr/bin/env python3
"""
SeniorTalk 测试集评估 - 使用镜像站版本
解决网络问题：使用 HF 镜像站下载模型
"""

import os
import sys
from pathlib import Path

# ============================================================================
# 重要：配置镜像站和缓存路径（必须在最开始）
# ============================================================================

# 获取用户主目录
HOME = Path.home()
PROJECT_ROOT = HOME / "AD_predict" / "AD_predict"

# 设置 Hugging Face 镜像站（解决网络问题）
# 方案1: hf-mirror.com (推荐)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 如果上面不行，可以尝试方案2: ModelScope (阿里云)
# os.environ['HF_ENDPOINT'] = 'https://www.modelscope.cn'

# 设置所有缓存到用户主目录
HF_CACHE_DIR = HOME / ".cache" / "huggingface"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(HF_CACHE_DIR)
os.environ['TRANSFORMERS_CACHE'] = str(HF_CACHE_DIR / "transformers")
os.environ['HF_DATASETS_CACHE'] = str(HF_CACHE_DIR / "datasets")
os.environ['HF_HUB_CACHE'] = str(HF_CACHE_DIR / "hub")

print(f"{'='*80}")
print(f"🌐 网络配置")
print(f"{'='*80}")
print(f"HF Mirror: {os.environ.get('HF_ENDPOINT', 'Default')}")
print(f"Cache Dir: {HF_CACHE_DIR}")
print(f"{'='*80}\n")

# 现在导入其他库
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import pandas as pd
import numpy as np
from tqdm import tqdm
import jiwer
from pypinyin import pinyin, Style
import re
import json
import glob
import soundfile as sf
import io
from datetime import datetime
import warnings
import logging

# 尝试导入多个音频解码库
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    warnings.warn("torchaudio not available, falling back to soundfile")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# 配置
# ==============================================================================

# 可用模型配置 - 使用完整路径
MODEL_CONFIGS = {
    "whisper_medium_baseline": {
        "base_model": "openai/whisper-medium",
        "adapter_path": None,
        "name": "Whisper-Medium (原始基线)",
        "type": "baseline"
    },
    "exp1_high_rank": {
        "base_model": "openai/whisper-medium",
        "adapter_path": "/data/AD_predict/exp1_high_rank/checkpoint-100",
        "name": "Exp1: High Rank (ckpt-100)",
        "type": "finetuned"
    },
    "exp2_low_lr": {
        "base_model": "openai/whisper-medium",
        "adapter_path": "/data/AD_predict/exp2_low_lr/checkpoint-1100",
        "name": "Exp2: Low LR (ckpt-1100)",
        "type": "finetuned"
    },
    "exp3_large_batch": {
        "base_model": "openai/whisper-medium",
        "adapter_path": "/data/AD_predict/exp3_large_batch/checkpoint-750",
        "name": "Exp3: Large Batch (ckpt-750)",
        "type": "finetuned"
    },
    "exp4_aggressive": {
        "base_model": "openai/whisper-medium",
        "adapter_path": "/data/AD_predict/exp4_aggressive/checkpoint-500",
        "name": "Exp4: Aggressive LR (ckpt-500)",
        "type": "finetuned"
    },
    "best_model": {
        "base_model": "openai/whisper-medium",
        "adapter_path": str(PROJECT_ROOT / "models" / "best_model"),
        "name": "Best Model (保存的最优)",
        "type": "finetuned"
    },
    "dialect_final": {
        "base_model": "openai/whisper-large-v3",
        "adapter_path": str(PROJECT_ROOT / "whisper_lora_dialect" / "final_adapter"),
        "name": "Dialect Final (Whisper-Large-v3)",
        "type": "finetuned"
    }
}

# 数据和输出路径
SENIORTALK_TEST_DIR = Path("/data/AD_predict/data/raw/seniortalk_full/sentence_data")
OUTPUT_DIR = HOME / "AD_predict_results" / "seniortalk_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 评估参数
NUM_TEST_SAMPLES = 100  # 每个模型测试的样本数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"{'='*80}")
print(f"SeniorTalk Evaluation - Mirror Version")
print(f"{'='*80}")
print(f"Device: {DEVICE}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"Cache Directory: {HF_CACHE_DIR}")
print(f"\nAudio Decoding Backends:")
print(f"  - torchaudio: {'✓ Available' if TORCHAUDIO_AVAILABLE else '✗ Not available'}")
print(f"  - soundfile: ✓ Available (v{sf.__version__})")
print(f"  - librosa: {'✓ Available' if LIBROSA_AVAILABLE else '✗ Not available'}")
print()

# ==============================================================================
# 音频解码函数（支持多种后端）
# ==============================================================================

def decode_audio_bytes(audio_bytes, sample_id=None):
    """
    使用多种方法尝试解码音频字节流
    
    Args:
        audio_bytes: 音频字节数据
        sample_id: 样本ID（用于错误日志）
    
    Returns:
        tuple: (audio_array, sample_rate) 或 (None, None) 如果失败
    """
    errors = []
    
    # 方法 1: torchaudio (推荐，最可靠)
    if TORCHAUDIO_AVAILABLE:
        try:
            audio_tensor, sr = torchaudio.load(io.BytesIO(audio_bytes))
            # 转换为 numpy 并确保单声道
            audio_array = audio_tensor.numpy()
            if audio_array.shape[0] > 1:  # 多声道
                audio_array = audio_array.mean(axis=0)
            else:
                audio_array = audio_array[0]
            return audio_array, sr
        except Exception as e:
            errors.append(f"torchaudio: {str(e)[:50]}")
    
    # 方法 2: soundfile
    try:
        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
        # 确保是单声道
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        return audio_array, sr
    except Exception as e:
        errors.append(f"soundfile: {str(e)[:50]}")
    
    # 方法 3: librosa (最慢但最兼容)
    if LIBROSA_AVAILABLE:
        try:
            audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
            return audio_array, sr
        except Exception as e:
            errors.append(f"librosa: {str(e)[:50]}")
    
    # 所有方法都失败
    error_msg = "; ".join(errors)
    if sample_id is not None:
        logger.warning(f"Failed to decode audio (sample {sample_id}): {error_msg}")
    
    return None, None

# ==============================================================================
# 数据加载函数（使用手动 Parquet 加载）
# ==============================================================================

def load_seniortalk_data(parquet_files, max_samples=None):
    """
    从 Parquet 文件加载 SeniorTalk 数据的生成器
    
    Args:
        parquet_files: Parquet 文件路径列表
        max_samples: 最大加载样本数（None 表示全部加载）
    
    Yields:
        dict: {'array': audio_array, 'sampling_rate': sr, 'reference': text}
    """
    sample_count = 0
    failed_count = 0
    total_attempted = 0
    
    for parquet_file in sorted(parquet_files):
        print(f"  Loading {Path(parquet_file).name}...")
        
        try:
            df = pd.read_parquet(parquet_file)
            print(f"    Found {len(df)} samples")
            
            for idx, row in df.iterrows():
                if max_samples and sample_count >= max_samples:
                    break
                
                total_attempted += 1
                
                try:
                    # 从 Parquet 提取音频和文本
                    if hasattr(row, 'audio'):
                        if isinstance(row.audio, dict) and 'bytes' in row.audio:
                            audio_bytes = row.audio['bytes']
                        elif isinstance(row.audio, bytes):
                            audio_bytes = row.audio
                        else:
                            audio_bytes = bytes(row.audio)
                    else:
                        logger.debug(f"    Warning: No audio column found in row {idx}")
                        failed_count += 1
                        continue
                    
                    # 获取文本
                    text = getattr(row, 'text', None) or getattr(row, 'sentence', '')
                    
                    # 使用多后端解码音频
                    audio_array, sr = decode_audio_bytes(audio_bytes, sample_id=total_attempted)
                    
                    # 检查解码是否成功
                    if audio_array is None or sr is None:
                        failed_count += 1
                        continue
                    
                    # 验证音频数据有效性
                    if len(audio_array) == 0:
                        logger.warning(f"    Empty audio array for sample {total_attempted}")
                        failed_count += 1
                        continue
                    
                    sample_count += 1
                    
                    yield {
                        'array': audio_array,
                        'sampling_rate': sr,
                        'reference': str(text)
                    }
                    
                except Exception as e:
                    logger.error(f"    Error loading sample {idx}: {e}")
                    failed_count += 1
                    continue
                    
        except Exception as e:
            print(f"  Error reading {parquet_file}: {e}")
            continue
    
    # 打印统计信息
    print(f"\n  Data Loading Statistics:")
    print(f"    - Successfully loaded: {sample_count}")
    print(f"    - Failed to decode: {failed_count}")
    print(f"    - Total attempted: {total_attempted}")
    if total_attempted > 0:
        success_rate = (sample_count / total_attempted) * 100
        print(f"    - Success rate: {success_rate:.1f}%")

# ==============================================================================
# 评估指标函数
# ==============================================================================

def clean_text(text):
    """清洗文本"""
    return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', str(text)).strip()

def get_pinyin_with_tone(text):
    """带声调拼音"""
    return [p[0] for p in pinyin(text, style=Style.TONE3, errors='ignore')]

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
    print(f"  Base model: {config['base_model']}")
    print(f"  Using mirror: {os.environ.get('HF_ENDPOINT', 'Default')}")
    
    try:
        processor = WhisperProcessor.from_pretrained(config['base_model'])
        model = WhisperForConditionalGeneration.from_pretrained(
            config['base_model'],
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
        
        if config['adapter_path'] and os.path.exists(config['adapter_path']):
            print(f"  + Adapter: {config['adapter_path']}")
            model = PeftModel.from_pretrained(model, config['adapter_path'])
            model = model.merge_and_unload()
        
        model.to(DEVICE).eval()
        print(f"  ✓ Loaded successfully")
        return processor, model
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def evaluate_model(model_key, config, data_generator):
    """评估单个模型"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {config['name']}")
    print(f"{'='*80}")
    
    processor, model = load_model(config)
    if not model:
        return None
    
    print(f"Testing {NUM_TEST_SAMPLES} samples...")
    
    results = []
    wer_scores, cer_scores, char_acc_scores, tone_acc_scores = [], [], [], []
    
    for idx, sample in enumerate(tqdm(data_generator, total=NUM_TEST_SAMPLES, desc="Evaluating")):
        try:
            audio_array = sample['array']
            sampling_rate = sample['sampling_rate']
            reference = sample['reference']
            
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
                "sample_id": idx,
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": float(wer),
                "cer": float(cer),
                "char_accuracy": float(char_acc),
                "tone_accuracy": float(tone_acc),
                "correct_chars": correct_chars
            })
            
        except Exception as e:
            print(f"\n  Error on sample {idx}: {e}")
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
    
    print(f"\n  WER: {summary['avg_wer']:.4f} (±{summary['std_wer']:.4f})")
    print(f"  CER: {summary['avg_cer']:.4f} (±{summary['std_cer']:.4f})")
    print(f"  Char Acc: {summary['avg_char_accuracy']:.4f}")
    print(f"  Tone Acc: {summary['avg_tone_accuracy']:.4f}")
    
    # 清理
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {"summary": summary, "details": results}

# ==============================================================================
# 主函数
# ==============================================================================

def main():
    print("="*80)
    print("Step 1: Loading SeniorTalk test dataset")
    print("="*80)
    
    # 查找 Parquet 文件
    parquet_pattern = str(SENIORTALK_TEST_DIR / "test-*.parquet")
    parquet_files = glob.glob(parquet_pattern)
    
    if not parquet_files:
        print(f"  ✗ No parquet files found at: {parquet_pattern}")
        return
    
    print(f"  Found {len(parquet_files)} parquet files:")
    for f in parquet_files:
        print(f"    - {Path(f).name}")
    
    # 检查可用模型
    print(f"\n{'='*80}")
    print("Step 2: Checking available models")
    print(f"{'='*80}")
    
    available = {}
    for key, cfg in MODEL_CONFIGS.items():
        if cfg['adapter_path'] is None or os.path.exists(cfg['adapter_path']):
            available[key] = cfg
            status = "baseline" if cfg['adapter_path'] is None else "✓"
            print(f"  {status} {cfg['name']}")
        else:
            print(f"  ✗ {cfg['name']} (not found at {cfg['adapter_path']})")
    
    if not available:
        print("\n✗ No models available!")
        return
    
    print(f"\n  Total available: {len(available)} models")
    
    # 评估所有模型
    print(f"\n{'='*80}")
    print("Step 3: Evaluating models")
    print(f"{'='*80}")
    
    all_results = {}
    
    for model_key in available:
        # 为每个模型创建新的数据生成器
        data_gen = load_seniortalk_data(parquet_files, max_samples=NUM_TEST_SAMPLES)
        
        result = evaluate_model(model_key, available[model_key], data_gen)
        
        if result:
            all_results[model_key] = result
            
            # 保存单个模型结果
            detail_file = OUTPUT_DIR / f"{model_key}_results.json"
            with open(detail_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {detail_file}")
    
    # 生成对比报告
    if all_results:
        print(f"\n{'='*80}")
        print("Step 4: Generating Comparison Report")
        print(f"{'='*80}")
        
        summaries = [r['summary'] for r in all_results.values()]
        df = pd.DataFrame(summaries)
        df = df.sort_values('avg_cer')
        
        # 保存汇总
        summary_file = OUTPUT_DIR / "evaluation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "num_samples": NUM_TEST_SAMPLES,
                "device": DEVICE,
                "mirror": os.environ.get('HF_ENDPOINT', 'Default'),
                "output_directory": str(OUTPUT_DIR),
                "models": summaries
            }, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Summary: {summary_file}")
        
        # 保存CSV
        csv_file = OUTPUT_DIR / "comparison_table.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"  ✓ CSV: {csv_file}")
        
        # 生成Markdown表格
        md_file = OUTPUT_DIR / "comparison_table.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# SeniorTalk Evaluation Results\n\n")
            f.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Test Samples:** {NUM_TEST_SAMPLES}\n\n")
            f.write(f"**Mirror:** {os.environ.get('HF_ENDPOINT', 'Default')}\n\n")
            f.write(f"**Output Directory:** {OUTPUT_DIR}\n\n")
            f.write("## Performance Comparison (Sorted by CER)\n\n")
            f.write(df[['model_name', 'avg_wer', 'avg_cer', 'avg_char_accuracy', 'avg_tone_accuracy']].to_markdown(index=False))
        print(f"  ✓ Markdown: {md_file}")
        
        # 打印表格
        print(f"\n{'='*80}")
        print("Performance Comparison")
        print(f"{'='*80}")
        print(df[['model_name', 'avg_wer', 'avg_cer', 'avg_char_accuracy', 'avg_tone_accuracy']].to_string(index=False))
        
        print(f"\n{'='*80}")
        print("✓ Evaluation Complete!")
        print(f"{'='*80}")
        print(f"\nResults saved to: {OUTPUT_DIR}/")
        print(f"  - evaluation_summary.json")
        print(f"  - comparison_table.csv")
        print(f"  - comparison_table.md")
        print(f"  - *_results.json (per-model details)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()





