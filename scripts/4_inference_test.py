"""
Whisper LoRA模型推理测试脚本
用于测试训练好的模型在新音频上的表现
"""

import os
import sys
import torch
import librosa
import argparse
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel


def load_model(base_model_name, adapter_path, device="cuda"):
    """
    加载Whisper基础模型和LoRA适配器
    
    Args:
        base_model_name: 基础模型名称
        adapter_path: LoRA适配器路径
        device: 设备（cuda或cpu）
        
    Returns:
        model, processor
    """
    print(f"加载基础模型: {base_model_name}")
    processor = WhisperProcessor.from_pretrained(
        base_model_name, 
        language="zh", 
        task="transcribe"
    )
    
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    print(f"加载LoRA适配器: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)
    model.eval()
    
    return model, processor


def transcribe_audio(audio_path, model, processor, dialect=None, device="cuda"):
    """
    对单个音频文件进行转录
    
    Args:
        audio_path: 音频文件路径
        model: Whisper模型
        processor: Whisper处理器
        dialect: 方言标签（可选）
        device: 设备
        
    Returns:
        转录文本
    """
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # 提取特征
    input_features = processor(
        audio, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features
    
    input_features = input_features.to(device, torch.float16 if device == "cuda" else torch.float32)
    
    # 准备decoder输入
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")
    
    # 如果指定了方言，可以在这里添加方言提示（需要进一步实现）
    if dialect:
        print(f"使用方言提示: {dialect}")
        # 注意：方言提示需要在训练时正确处理token
    
    # 生成转录
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_length=225
        )
    
    # 解码
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription


def batch_transcribe(audio_dir, model, processor, output_file=None, dialect=None, device="cuda"):
    """
    批量转录目录中的所有音频文件
    
    Args:
        audio_dir: 音频目录
        model: Whisper模型
        processor: Whisper处理器
        output_file: 输出文件路径（可选）
        dialect: 方言标签（可选）
        device: 设备
    """
    import glob
    
    # 查找所有音频文件
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
        audio_files.extend(glob.glob(os.path.join(audio_dir, ext)))
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    results = []
    
    for audio_file in audio_files:
        print(f"\n处理: {audio_file}")
        try:
            transcription = transcribe_audio(audio_file, model, processor, dialect, device)
            print(f"转录: {transcription}")
            results.append({
                "file": os.path.basename(audio_file),
                "transcription": transcription
            })
        except Exception as e:
            print(f"错误: {e}")
            results.append({
                "file": os.path.basename(audio_file),
                "error": str(e)
            })
    
    # 保存结果
    if output_file:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Whisper LoRA模型推理测试")
    parser.add_argument(
        "--audio",
        type=str,
        help="音频文件路径或目录"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="./whisper_lora_dialect/final_adapter",
        help="LoRA适配器路径"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="openai/whisper-large-v3",
        help="基础模型名称"
    )
    parser.add_argument(
        "--dialect",
        type=str,
        choices=[
            "beijing_mandarin", "wu_dialect", "dongbei_mandarin",
            "zhongyuan_mandarin", "lanyin_mandarin", "jianghuai_mandarin",
            "xinan_mandarin", "jin_dialect", "yue_dialect",
            "gan_dialect", "min_dialect", "tibetan_dialect"
        ],
        help="方言标签（可选）"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出文件路径（用于批量处理）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="计算设备"
    )
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.audio:
        print("错误: 请指定音频文件或目录 (--audio)")
        return
    
    if not os.path.exists(args.adapter):
        print(f"错误: LoRA适配器不存在: {args.adapter}")
        print("请先运行训练脚本 2_finetune_whisper_lora.py")
        return
    
    print("=" * 80)
    print("Whisper LoRA 模型推理")
    print("=" * 80)
    
    # 加载模型
    model, processor = load_model(args.base_model, args.adapter, args.device)
    
    # 推理
    if os.path.isfile(args.audio):
        # 单文件推理
        print(f"\n转录音频: {args.audio}")
        transcription = transcribe_audio(
            args.audio, 
            model, 
            processor, 
            args.dialect, 
            args.device
        )
        print(f"\n转录结果: {transcription}")
    
    elif os.path.isdir(args.audio):
        # 批量推理
        print(f"\n批量转录目录: {args.audio}")
        results = batch_transcribe(
            args.audio, 
            model, 
            processor, 
            args.output, 
            args.dialect, 
            args.device
        )
        print(f"\n共处理 {len(results)} 个文件")
    
    else:
        print(f"错误: 音频路径不存在: {args.audio}")
    
    print("\n" + "=" * 80)
    print("推理完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

