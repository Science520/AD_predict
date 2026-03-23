#!/usr/bin/env python3
"""
CPU评估脚本 - 在CPU上评估Whisper微调模型
适用于GPU被占用的情况
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm
import jiwer
from pypinyin import pinyin, Style
import re
import os
import sys
import soundfile as sf
import librosa
import numpy as np

# 强制使用CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ==============================================================================
# 配置区域
# ==============================================================================

# 模型配置（为了快速测试，只评估最重要的3个模型）
MODEL_CONFIGS = {
    "whisper_medium_base": {
        "base_model": "openai/whisper-medium",
        "path": None,
        "name": "原始Whisper-Medium"
    },
    "exp1_high_rank_100": {
        "base_model": "openai/whisper-medium",
        "path": "/data/AD_predict/exp1_high_rank/checkpoint-100/",
        "name": "Exp1: 高Rank (最佳)"
    },
    "exp3_large_batch_750": {
        "base_model": "openai/whisper-medium",
        "path": "/data/AD_predict/exp3_large_batch/checkpoint-750/",
        "name": "Exp3: 大Batch (最佳)"
    },
}

PROCESSED_DATA_DIR = "/data/AD_predict/processed_data"
NUM_TEST_SAMPLES = 30  # CPU较慢，先测试30个样本

# ==============================================================================
# 评估指标
# ==============================================================================

def clean_text(text):
    """清洗文本"""
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    return text.strip()

def get_pinyin_list(text):
    """转换为拼音列表"""
    return [p[0] for p in pinyin(text, style=Style.TONE3, errors='ignore')]

def compute_cer(reference, hypothesis):
    """计算字错误率"""
    ref_clean = clean_text(reference)
    hyp_clean = clean_text(hypothesis)
    if not ref_clean:
        return 1.0
    return jiwer.cer(ref_clean, hyp_clean)

def compute_pinyin_accuracy(reference, hypothesis):
    """计算拼音准确率"""
    ref_clean = clean_text(reference)
    hyp_clean = clean_text(hypothesis)
    
    ref_pinyin = get_pinyin_list(ref_clean)
    hyp_pinyin = get_pinyin_list(hyp_clean)
    
    if not ref_pinyin:
        return 0.0
    
    # jiwer.wer需要字符串列表，将拼音列表转为字符串
    ref_pinyin_str = ' '.join(ref_pinyin)
    hyp_pinyin_str = ' '.join(hyp_pinyin)
    
    try:
        pinyin_error_rate = jiwer.wer(ref_pinyin_str, hyp_pinyin_str)
        return 1.0 - pinyin_error_rate
    except:
        # 如果计算失败，使用基于字符的相似度
        return 1.0 - jiwer.cer(ref_pinyin_str, hyp_pinyin_str)

# ==============================================================================
# 音频加载
# ==============================================================================

def load_audio_from_path(audio_path):
    """加载音频文件"""
    if not os.path.isabs(audio_path) and not os.path.exists(audio_path):
        base_dirs = [
            '/data/AD_predict/data/raw/audio/elderly_audios',
            '/data/AD_predict/data/raw/audio/elderly_audios_augmented',
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
        audio_array = np.zeros(16000)
        sample_rate = 16000
    
    return audio_array, sample_rate

# ==============================================================================
# 主评估流程
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

def evaluate_model(model, processor, dataset, model_name):
    """评估单个模型（CPU版本，仅标准模式）"""
    model.eval()
    
    results = {
        "cer_standard": [],
        "pinyin_acc_standard": [],
        "transcriptions": []
    }
    
    # 直接从Arrow表提取数据，避免触发audio解码
    # 获取底层的Arrow表
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
        ).input_features
        
        with torch.no_grad():
            # 标准模式评估（贪心解码）
            predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            cer = compute_cer(reference_text, transcription)
            pinyin_acc = compute_pinyin_accuracy(reference_text, transcription)
            
            results["cer_standard"].append(cer)
            results["pinyin_acc_standard"].append(pinyin_acc)
            
            # 保存前5个示例
            if i < 5:
                results["transcriptions"].append({
                    "reference": reference_text,
                    "hypothesis": transcription,
                    "cer": cer,
                    "pinyin_acc": pinyin_acc
                })
    
    return results

def main():
    print("=" * 80)
    print("CPU评估脚本 - Whisper模型评估")
    print("=" * 80)
    print("⚠️  使用CPU运行，速度较慢，请耐心等待...")
    print(f"📊 将评估 {len(MODEL_CONFIGS)} 个模型，每个模型测试 {NUM_TEST_SAMPLES} 个样本")
    
    # 加载数据集
    print(f"\n加载验证集: {PROCESSED_DATA_DIR}")
    try:
        dataset_dict = load_from_disk(PROCESSED_DATA_DIR)
        val_dataset = dataset_dict['validation']
        
        if NUM_TEST_SAMPLES is not None:
            val_dataset = val_dataset.shuffle(seed=42).select(range(min(NUM_TEST_SAMPLES, len(val_dataset))))
        
        print(f"✓ 成功加载 {len(val_dataset)} 条验证样本")
    except Exception as e:
        print(f"❌ 错误: 无法加载数据集! {e}")
        return
    
    all_results = []
    
    # 遍历模型进行评估
    for model_id, config in MODEL_CONFIGS.items():
        print("\n" + "=" * 80)
        print(f"正在加载模型: {config['name']} ({model_id})")
        print("=" * 80)
        
        try:
            # 加载模型和处理器（在CPU上）
            print("⏳ 加载模型到CPU（这可能需要几分钟）...")
            processor = WhisperProcessor.from_pretrained(config['base_model'])
            model = WhisperForConditionalGeneration.from_pretrained(
                config['base_model'],
                torch_dtype=torch.float32  # CPU使用float32
            )
            
            if config['path']:
                adapter_config_path = os.path.join(config['path'], 'adapter_config.json')
                if not os.path.exists(adapter_config_path):
                    print(f"⚠️  警告: 在 {config['path']} 中未找到 'adapter_config.json'。跳过此模型。")
                    continue
                print(f"✓ 加载LoRA adapter from: {config['path']}")
                model = PeftModel.from_pretrained(model, config['path'])
            
            model.config.forced_decoder_ids = None
            print("✓ 模型加载完成")
            
            # 评估模型
            model_results = evaluate_model(
                model, 
                processor, 
                val_dataset, 
                config['name']
            )
            
            # 计算平均分
            avg_cer = sum(model_results["cer_standard"]) / len(model_results["cer_standard"])
            avg_pinyin = sum(model_results["pinyin_acc_standard"]) / len(model_results["pinyin_acc_standard"])
            
            all_results.append({
                "模型名称": config['name'],
                "模型ID": model_id,
                "字正确率(%)": (1 - avg_cer) * 100,
                "拼音准确率(%)": avg_pinyin * 100,
                "平均CER": avg_cer,
            })
            
            # 打印示例转录结果
            print(f"\n{config['name']} - 示例转录结果 (前3条):")
            for i, trans in enumerate(model_results["transcriptions"][:3]):
                print(f"\n样本 {i+1}:")
                print(f"  参考: {trans['reference'][:80]}")
                print(f"  预测: {trans['hypothesis'][:80]}")
                print(f"  字正确率: {(1 - trans['cer']) * 100:.2f}%")
                print(f"  拼音准确率: {trans['pinyin_acc'] * 100:.2f}%")
            
            # 清理内存
            del model
            del processor
            
        except Exception as e:
            print(f"❌ 评估模型 {config['name']} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 打印和保存最终结果
    if not all_results:
        print("\n❌ 没有成功评估任何模型！")
        return
    
    results_df = pd.DataFrame(all_results)
    
    print("\n\n" + "=" * 100)
    print("📊 CPU评估结果汇总")
    print("=" * 100)
    
    pd.options.display.float_format = '{:.2f}'.format
    results_df = results_df.sort_values('字正确率(%)', ascending=False)
    print(results_df[["模型名称", "字正确率(%)", "拼音准确率(%)", "平均CER"]].to_string(index=False))
    
    # 保存结果
    output_dir = "/data/AD_predict/all_experiments_20251022_140017"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cpu_evaluation_results.csv")
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 结果已保存到: {output_path}")
    
    # 保存简短的Markdown报告
    markdown_path = os.path.join(output_dir, "CPU_EVALUATION_REPORT.md")
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write("# 🖥️ CPU评估报告 - Whisper模型\n\n")
        f.write(f"**评估样本数**: {len(val_dataset)}\n")
        f.write(f"**评估设备**: CPU\n")
        f.write(f"**评估模式**: 标准模式（贪心解码）\n\n")
        f.write("---\n\n")
        f.write("## 📊 结果汇总\n\n")
        f.write(results_df[["模型名称", "字正确率(%)", "拼音准确率(%)", "平均CER"]].to_markdown(index=False))
        f.write("\n\n---\n\n")
        
        # 分析结果
        best_model = results_df.iloc[0]
        base_model = results_df[results_df['模型ID'] == 'whisper_medium_base']
        
        f.write("## 💡 关键发现\n\n")
        f.write(f"1. **最佳模型**: {best_model['模型名称']}\n")
        f.write(f"   - 字正确率: {best_model['字正确率(%)']:.2f}%\n")
        f.write(f"   - 拼音准确率: {best_model['拼音准确率(%)']:.2f}%\n\n")
        
        if not base_model.empty:
            base_acc = base_model.iloc[0]['字正确率(%)']
            best_acc = best_model['字正确率(%)']
            f.write(f"2. **原始模型表现**: {base_acc:.2f}%\n")
            if best_acc < base_acc:
                f.write(f"   - ⚠️ 微调后性能下降了 {base_acc - best_acc:.2f}%\n")
                f.write(f"   - 这证实了过拟合问题\n\n")
            else:
                f.write(f"   - ✅ 微调后性能提升了 {best_acc - base_acc:.2f}%\n\n")
        
        f.write("## 📝 说明\n\n")
        f.write("- 本次评估在CPU上运行，因GPU被占用\n")
        f.write("- 仅使用标准模式（贪心解码），未使用CI候选集模式\n")
        f.write("- 如需完整评估，请在GPU可用时运行GPU版本脚本\n")
    
    print(f"✅ 详细报告已保存到: {markdown_path}")
    
    print("\n" + "=" * 100)
    print("✅ CPU评估完成！")
    print("=" * 100)


if __name__ == "__main__":
    main()

