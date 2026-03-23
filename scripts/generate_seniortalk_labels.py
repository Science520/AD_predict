#!/usr/bin/env python3
"""
使用Whisper-large-v3为SeniorTalk数据生成伪标签
"""
import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def generate_pseudo_labels(
    manifest_path: str,
    output_path: str,
    model_name: str = "openai/whisper-large-v3",
    device: str = "cuda",
    batch_size: int = 1,
    language: str = "zh"
):
    """
    为音频数据生成伪标签
    
    Args:
        manifest_path: 输入清单文件路径（jsonl格式）
        output_path: 输出清单文件路径
        model_name: Whisper模型名称
        device: 计算设备 (cuda/cpu)
        batch_size: 批处理大小
        language: 语言代码
    """
    
    # 检查设备
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA不可用，切换到CPU")
        device = "cpu"
    
    print("="*80)
    print(f"🤖 使用 {model_name} 生成伪标签")
    print(f"🎮 设备: {device}")
    print(f"📦 批处理大小: {batch_size}")
    print("="*80)
    
    # 加载模型
    print("\n📥 加载模型...")
    try:
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        sys.exit(1)
    
    # 读取清单
    print(f"\n📋 读取清单: {manifest_path}")
    manifest_data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            manifest_data.append(json.loads(line))
    
    total_samples = len(manifest_data)
    print(f"✅ 读取 {total_samples} 条记录")
    
    # 处理每个样本
    print(f"\n🔄 开始生成转录...")
    updated_data = []
    failed_count = 0
    
    for idx, item in enumerate(tqdm(manifest_data, desc="生成转录")):
        audio_path = item['audio_path']
        
        try:
            # 检查文件是否存在
            if not os.path.exists(audio_path):
                print(f"\n⚠️ 文件不存在: {audio_path}")
                failed_count += 1
                continue
            
            # 加载音频
            audio_array, sr = librosa.load(audio_path, sr=16000)
            
            # 跳过太短的音频
            duration = len(audio_array) / sr
            if duration < 0.5:
                print(f"\n⚠️ 音频太短 ({duration:.2f}s): {audio_path}")
                failed_count += 1
                continue
            
            # 处理音频
            input_features = processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(device)
            
            # 生成转录
            with torch.no_grad():
                # 强制使用中文
                forced_decoder_ids = processor.get_decoder_prompt_ids(
                    language=language,
                    task="transcribe"
                )
                
                predicted_ids = model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_length=448,
                    num_beams=5,
                    temperature=0.0
                )
            
            # 解码
            transcription = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            # 更新数据
            item['text'] = transcription.strip()
            item['pseudo_label'] = True  # 标记为伪标签
            item['model_used'] = model_name
            updated_data.append(item)
            
            # 每100条保存一次（防止意外中断）
            if (idx + 1) % 100 == 0:
                temp_output = output_path.replace('.jsonl', f'_checkpoint_{idx+1}.jsonl')
                with open(temp_output, 'w', encoding='utf-8') as f:
                    for data_item in updated_data:
                        f.write(json.dumps(data_item, ensure_ascii=False) + '\n')
                print(f"\n💾 已保存检查点: {temp_output}")
            
        except Exception as e:
            print(f"\n❌ 处理失败 {audio_path}: {e}")
            failed_count += 1
            continue
    
    # 保存最终结果
    print(f"\n💾 保存最终结果: {output_path}")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in updated_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 统计信息
    print("\n" + "="*80)
    print("✅ 伪标签生成完成！")
    print("="*80)
    print(f"📊 统计:")
    print(f"   总样本数: {total_samples}")
    print(f"   成功: {len(updated_data)} ({len(updated_data)/total_samples*100:.1f}%)")
    print(f"   失败: {failed_count} ({failed_count/total_samples*100:.1f}%)")
    print(f"   输出文件: {output_path}")
    print("="*80)
    
    # 显示一些示例
    print("\n📝 转录示例（前5条）:")
    for i, item in enumerate(updated_data[:5]):
        print(f"\n{i+1}. 文件: {Path(item['audio_path']).name}")
        print(f"   时长: {item.get('duration', 'N/A'):.2f}s")
        print(f"   转录: {item['text']}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="为SeniorTalk数据生成伪标签")
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="输入清单文件路径 (jsonl格式)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出清单文件路径"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-large-v3",
        help="Whisper模型名称 (默认: openai/whisper-large-v3)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="计算设备 (默认: cuda)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批处理大小 (默认: 1)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        help="语言代码 (默认: zh)"
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.manifest):
        print(f"❌ 输入文件不存在: {args.manifest}")
        sys.exit(1)
    
    # 生成伪标签
    output_path = generate_pseudo_labels(
        manifest_path=args.manifest,
        output_path=args.output,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        language=args.language
    )
    
    print(f"\n🎉 完成！下一步:")
    print(f"1. 查看输出: {output_path}")
    print(f"2. 检查转录质量（随机抽查几条）")
    print(f"3. 运行数据划分脚本")
    print(f"4. 整合到训练pipeline")


if __name__ == "__main__":
    main()

