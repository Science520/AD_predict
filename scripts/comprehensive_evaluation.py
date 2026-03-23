#!/usr/bin/env python3
"""
综合评估脚本 - 评估Whisper微调模型
功能：
1. 加载5个模型：4个微调模型 + 1个原始模型
2. 使用验证集进行评估
3. 两种评估模式：标准模式 + CI候选集模式
4. 计算字错误率(CER)和拼音准确率
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import pandas as pd
from datasets import load_from_disk, Dataset
from tqdm import tqdm
import jiwer
from pypinyin import pinyin, Style
import re
import os
import sys
from pathlib import Path
import soundfile as sf
import librosa
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==============================================================================
# 1. 配置区域
# ==============================================================================

# 模型配置
MODEL_CONFIGS = {
    "whisper_medium_base": {
        "base_model": "openai/whisper-medium",
        "path": None,  # 不加载LoRA
        "name": "原始Whisper-Medium"
    },
    "exp1_high_rank_100": {
        "base_model": "openai/whisper-medium",
        "path": "/data/AD_predict/exp1_high_rank/checkpoint-100/",
        "name": "Exp1: 高Rank (step 100, 最佳)"
    },
    "exp2_low_lr_1100": {
        "base_model": "openai/whisper-medium",
        "path": "/data/AD_predict/exp2_low_lr/checkpoint-1100/",
        "name": "Exp2: 低学习率 (step 1100)"
    },
    "exp3_large_batch_750": {
        "base_model": "openai/whisper-medium",
        "path": "/data/AD_predict/exp3_large_batch/checkpoint-750/",
        "name": "Exp3: 大Batch (step 750, 最佳)"
    },
    "exp4_aggressive_500": {
        "base_model": "openai/whisper-medium",
        "path": "/data/AD_predict/exp4_aggressive/checkpoint-500/",
        "name": "Exp4: 激进学习率 (step 500)"
    },
}

# 数据配置
PROCESSED_DATA_DIR = "/data/AD_predict/processed_data"
NUM_TEST_SAMPLES = 30  # 测试30个样本

# 其他配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CANDIDATES = 5  # CI候选集数量
BATCH_SIZE = 1  # 推理batch size

# ==============================================================================
# 2. 评估指标计算函数
# ==============================================================================

def clean_text(text):
    """简单的文本清洗，去除标点和多余空格"""
    # 只保留中文字符、英文字母和数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    return text.strip()

def get_pinyin_list(text):
    """将文本转换为带声调的拼音列表"""
    # style=Style.TONE3 表示声调在末尾，如 'hǎo' -> 'hao3'
    # errors='ignore' 忽略无法转换的字符
    return [p[0] for p in pinyin(text, style=Style.TONE3, errors='ignore')]

def compute_cer(reference, hypothesis):
    """计算字错误率 (Character Error Rate)"""
    ref_clean = clean_text(reference)
    hyp_clean = clean_text(hypothesis)
    
    if not ref_clean:
        return 1.0  # 如果参考文本为空，错误率为1
    
    # jiwer.cer 计算字符级别的错误率
    return jiwer.cer(ref_clean, hyp_clean)

def compute_pinyin_accuracy(reference, hypothesis):
    """计算拼音正确率 (基于编辑距离)"""
    ref_clean = clean_text(reference)
    hyp_clean = clean_text(hypothesis)
    
    ref_pinyin = get_pinyin_list(ref_clean)
    hyp_pinyin = get_pinyin_list(hyp_clean)
    
    if not ref_pinyin:
        return 0.0  # 如果参考拼音为空，准确率为0

    # jiwer.wer需要字符串，将拼音列表转为字符串
    ref_pinyin_str = ' '.join(ref_pinyin)
    hyp_pinyin_str = ' '.join(hyp_pinyin)
    
    try:
        pinyin_error_rate = jiwer.wer(ref_pinyin_str, hyp_pinyin_str)
        return 1.0 - pinyin_error_rate
    except:
        # 如果计算失败，使用基于字符的相似度
        return 1.0 - jiwer.cer(ref_pinyin_str, hyp_pinyin_str)

# ==============================================================================
# 3. 音频加载辅助函数
# ==============================================================================

def load_audio_from_path(audio_path):
    """手动加载音频文件，支持绝对路径和相对路径"""
    # 尝试多个可能的基础目录
    if not os.path.isabs(audio_path) and not os.path.exists(audio_path):
        base_dirs = [
            '/data/AD_predict/data/raw/audio/elderly_audios',
            '/data/AD_predict/data/raw/audio/elderly_audios_augmented',
            os.path.expanduser('~/AD_predict/AD_predict/data/raw/audio/elderly_audios'),
            os.path.expanduser('~/AD_predict/AD_predict/data/raw/audio/elderly_audios_augmented'),
        ]
        for base_dir in base_dirs:
            full_path = os.path.join(base_dir, audio_path)
            if os.path.exists(full_path):
                audio_path = full_path
                break
    
    try:
        audio_array, sample_rate = sf.read(audio_path)
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        if sample_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
    except Exception as e:
        print(f"警告: 无法加载音频 {audio_path}: {e}")
        # 返回静音
        audio_array = np.zeros(16000)
        sample_rate = 16000
    
    return audio_array, sample_rate

# ==============================================================================
# 4. 主评估流程
# ==============================================================================

def extract_actual_text(text_with_tokens):
    """从包含Whisper特殊token的文本中提取实际转录内容"""
    # 找到最后一个特殊token (以|>结尾) 之后的内容
    if "<|dialect:" in text_with_tokens:
        # 分割并取最后一个>之后的内容
        actual_text = text_with_tokens.split(">")[-1].strip()
        return actual_text
    # 如果没有特殊token，直接返回
    return text_with_tokens

def evaluate_model(model, processor, dataset, model_name, device):
    """评估单个模型"""
    model.eval()
    
    results = {
        "cer_standard": [],
        "pinyin_acc_standard": [],
        "cer_ci_oracle": [],
        "pinyin_acc_ci_oracle": [],
        "transcriptions": []  # 保存一些示例转录结果
    }
    
    # 直接从Arrow表提取数据，避免触发audio解码
    arrow_table = dataset._data
    
    # 遍历测试数据
    for i in tqdm(range(len(dataset)), desc=f"评估 {model_name}"):
        # 直接从Arrow表提取path，避免解码音频
        audio_dict = arrow_table['audio'][i].as_py()
        audio_path = audio_dict['path'] if isinstance(audio_dict, dict) else str(audio_dict)
        text_with_tokens = arrow_table['text'][i].as_py()
        
        # 提取实际转录文本（去掉Whisper特殊token）
        reference_text = extract_actual_text(text_with_tokens)
        
        audio_array, sample_rate = load_audio_from_path(audio_path)
        
        # 预处理音频
        input_features = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device)
        
        with torch.no_grad():
            # --- 标准模式评估 ---
            predicted_ids = model.generate(input_features)
            transcription_standard = processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            cer_std = compute_cer(reference_text, transcription_standard)
            pinyin_acc_std = compute_pinyin_accuracy(reference_text, transcription_standard)
            
            results["cer_standard"].append(cer_std)
            results["pinyin_acc_standard"].append(pinyin_acc_std)
            
            # --- CI候选集模式评估 ---
            try:
                predicted_ids_ci = model.generate(
                    input_features,
                    num_beams=5,
                    num_return_sequences=NUM_CANDIDATES,
                    early_stopping=True
                )
                transcriptions_ci = processor.batch_decode(
                    predicted_ids_ci, 
                    skip_special_tokens=True
                )
                
                # 找出候选集中最好的一个（Oracle结果）
                best_cer = min([compute_cer(reference_text, cand) for cand in transcriptions_ci])
                best_pinyin_acc = max([compute_pinyin_accuracy(reference_text, cand) for cand in transcriptions_ci])
                
                results["cer_ci_oracle"].append(best_cer)
                results["pinyin_acc_ci_oracle"].append(best_pinyin_acc)
            except Exception as e:
                # 如果CI模式失败，使用标准模式结果
                results["cer_ci_oracle"].append(cer_std)
                results["pinyin_acc_ci_oracle"].append(pinyin_acc_std)
            
            # 保存前5个示例
            if i < 5:
                results["transcriptions"].append({
                    "reference": reference_text,
                    "hypothesis": transcription_standard,
                    "cer": cer_std,
                    "pinyin_acc": pinyin_acc_std
                })
    
    return results

def main():
    print("=" * 80)
    print("开始综合评估")
    print("=" * 80)
    print(f"使用设备: {DEVICE}")
    
    # --- 加载数据集 ---
    print(f"\n加载验证集: {PROCESSED_DATA_DIR}")
    try:
        from datasets import load_from_disk
        dataset_dict = load_from_disk(PROCESSED_DATA_DIR)
        val_dataset = dataset_dict['validation']
        
        if NUM_TEST_SAMPLES is not None:
            val_dataset = val_dataset.shuffle(seed=42).select(range(min(NUM_TEST_SAMPLES, len(val_dataset))))
        
        print(f"成功加载 {len(val_dataset)} 条验证样本")
    except Exception as e:
        print(f"错误: 无法加载数据集! {e}")
        return
    
    all_results = []
    
    # --- 遍历模型进行评估 ---
    for model_id, config in MODEL_CONFIGS.items():
        print("\n" + "=" * 80)
        print(f"正在加载模型: {config['name']} ({model_id})")
        print("=" * 80)
        
        # --- 加载模型和处理器 ---
        try:
            processor = WhisperProcessor.from_pretrained(config['base_model'])
            model = WhisperForConditionalGeneration.from_pretrained(
                config['base_model']
            ).to(DEVICE)
            
            if config['path']:
                # 检查adapter_config.json是否存在
                adapter_config_path = os.path.join(config['path'], 'adapter_config.json')
                if not os.path.exists(adapter_config_path):
                    print(f"⚠️  警告: 在 {config['path']} 中未找到 'adapter_config.json'。跳过此模型。")
                    continue
                print(f"✓ 加载LoRA adapter from: {config['path']}")
                model = PeftModel.from_pretrained(model, config['path'])
            
            model.config.forced_decoder_ids = None
            
            # 评估模型
            model_results = evaluate_model(
                model, 
                processor, 
                val_dataset, 
                config['name'],
                DEVICE
            )
            
            # 计算平均分
            avg_cer_std = sum(model_results["cer_standard"]) / len(model_results["cer_standard"])
            avg_pinyin_std = sum(model_results["pinyin_acc_standard"]) / len(model_results["pinyin_acc_standard"])
            avg_cer_ci = sum(model_results["cer_ci_oracle"]) / len(model_results["cer_ci_oracle"])
            avg_pinyin_ci = sum(model_results["pinyin_acc_ci_oracle"]) / len(model_results["pinyin_acc_ci_oracle"])
            
            all_results.append({
                "模型名称": config['name'],
                "模型ID": model_id,
                "标准模式-字正确率(%)": (1 - avg_cer_std) * 100,
                "标准模式-拼音准确率(%)": avg_pinyin_std * 100,
                "CI模式-最佳字正确率(%)": (1 - avg_cer_ci) * 100,
                "CI模式-最佳拼音准确率(%)": avg_pinyin_ci * 100,
            })
            
            # 打印一些示例转录结果
            print(f"\n{config['name']} - 示例转录结果 (前3条):")
            for i, trans in enumerate(model_results["transcriptions"][:3]):
                print(f"\n样本 {i+1}:")
                print(f"  参考: {trans['reference'][:80]}")
                print(f"  预测: {trans['hypothesis'][:80]}")
                print(f"  字正确率: {(1 - trans['cer']) * 100:.2f}%")
                print(f"  拼音准确率: {trans['pinyin_acc'] * 100:.2f}%")
            
            # 清理GPU内存
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ 评估模型 {config['name']} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # --- 打印和保存最终结果 ---
    if not all_results:
        print("\n❌ 没有成功评估任何模型！")
        return
    
    results_df = pd.DataFrame(all_results)
    
    print("\n\n" + "=" * 100)
    print("📊 评估结果汇总")
    print("=" * 100)
    
    # 格式化输出，保留两位小数
    pd.options.display.float_format = '{:.2f}'.format
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    
    # 按标准模式字正确率排序
    results_df = results_df.sort_values('标准模式-字正确率(%)', ascending=False)
    print(results_df.to_string(index=False))
    
    # 保存到CSV
    output_dir = "/data/AD_predict/all_experiments_20251022_140017"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "comprehensive_evaluation_results.csv")
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 结果已保存到: {output_path}")
    
    # 保存详细的Markdown报告
    markdown_path = os.path.join(output_dir, "EVALUATION_REPORT.md")
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write("# 🎯 Whisper模型综合评估报告\n\n")
        f.write(f"**评估样本数**: {len(val_dataset)}\n")
        f.write(f"**评估模式**: 标准模式 + CI候选集模式 (5个候选)\n")
        f.write(f"**评估指标**: 字正确率 (1-CER) + 拼音准确率\n\n")
        f.write("---\n\n")
        f.write("## 📊 结果汇总\n\n")
        
        # 手动创建Markdown表格（避免依赖tabulate）
        f.write("| 模型名称 | 模型ID | 标准-字正确率(%) | 标准-拼音准确率(%) | CI-最佳字正确率(%) | CI-最佳拼音准确率(%) |\n")
        f.write("|---------|--------|-----------------|-------------------|------------------|--------------------|\n")
        for _, row in results_df.iterrows():
            f.write(f"| {row['模型名称']} | {row['模型ID']} | {row['标准模式-字正确率(%)']:.2f} | {row['标准模式-拼音准确率(%)']:.2f} | {row['CI模式-最佳字正确率(%)']:.2f} | {row['CI模式-最佳拼音准确率(%)']:.2f} |\n")
        
        f.write("\n\n---\n\n")
        f.write("## 💡 关键发现\n\n")
        
        # 分析结果
        best_model = results_df.iloc[0]
        worst_model = results_df.iloc[-1]
        base_model = results_df[results_df['模型ID'] == 'whisper_medium_base']
        
        f.write(f"1. **最佳模型**: {best_model['模型名称']}\n")
        f.write(f"   - 标准模式字正确率: {best_model['标准模式-字正确率(%)']:.2f}%\n")
        f.write(f"   - CI模式最佳字正确率: {best_model['CI模式-最佳字正确率(%)']:.2f}%\n\n")
        
        if not base_model.empty:
            base_acc = base_model.iloc[0]['标准模式-字正确率(%)']
            best_acc = best_model['标准模式-字正确率(%)']
            f.write(f"2. **原始模型表现**: {base_acc:.2f}%\n")
            if best_acc < base_acc:
                f.write(f"   - ⚠️ 微调后性能下降了 {base_acc - best_acc:.2f}%\n")
                f.write(f"   - 可能原因: 数据量不足、过拟合、领域不匹配\n\n")
            else:
                f.write(f"   - ✅ 微调后性能提升了 {best_acc - base_acc:.2f}%\n\n")
        
        f.write(f"3. **CI模式改进空间**:\n")
        for _, row in results_df.iterrows():
            ci_gain = row['CI模式-最佳字正确率(%)'] - row['标准模式-字正确率(%)']
            f.write(f"   - {row['模型名称']}: +{ci_gain:.2f}%\n")
        
        f.write("\n---\n\n")
        f.write("## 📝 说明\n\n")
        f.write("- **字正确率**: `(1 - CER) × 100%`，CER为字错误率\n")
        f.write("- **拼音准确率**: 基于拼音序列编辑距离的准确率，衡量语音相似度\n")
        f.write("- **标准模式**: 使用贪心解码生成最优转录\n")
        f.write("- **CI模式**: 生成5个候选，选择最佳的一个（Oracle）\n")
        f.write("- **CI模式改进空间**: 表示如果有完美的重排序器，能提升多少性能\n")
    
    print(f"✅ 详细报告已保存到: {markdown_path}")
    
    print("\n" + "=" * 100)
    print("✅ 评估完成！")
    print("=" * 100)


if __name__ == "__main__":
    main()

