#!/usr/bin/env python3
"""
整合SeniorTalk ASR数据到现有训练pipeline
"""
import os
import sys
import json
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import librosa
import soundfile as sf

def extract_seniortalk_tar(tar_path: str, output_dir: str, max_samples: int = None):
    """
    从tar包中提取音频文件
    
    Args:
        tar_path: tar包路径
        output_dir: 输出目录
        max_samples: 最大提取样本数（None表示全部提取）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 开始解压: {tar_path}")
    print(f"📁 输出到: {output_dir}")
    
    extracted_files = []
    
    with tarfile.open(tar_path, 'r') as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith('.wav')]
        
        if max_samples:
            members = members[:max_samples]
        
        print(f"🎵 总共{len(members)}个音频文件")
        
        for member in tqdm(members, desc="解压音频"):
            # 提取文件
            tar.extract(member, output_path)
            extracted_path = output_path / member.name
            extracted_files.append(str(extracted_path))
    
    print(f"✅ 解压完成: {len(extracted_files)}个文件")
    return extracted_files


def parse_utterance_info(utterance_info_path: str):
    """
    解析UTTERANCEINFO.txt获取说话人和方言信息
    
    返回: dict {speaker_id: {dialect, age, gender, ...}}
    """
    print(f"📋 解析说话人信息: {utterance_info_path}")
    
    speaker_info = {}
    
    with open(utterance_info_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # 跳过header
        
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 7:
                filename = parts[0]
                speaker1_id = parts[1]
                speaker2_id = parts[2]
                
                # 从文件名提取方言信息
                # 格式: 001&002-81&78-MF-BEIJING&SHANDONG-HUAWEI MGA-AL00.wav
                try:
                    dialect_part = parts[0].split('-')[3]
                    dialects = dialect_part.split('&')
                    
                    if speaker1_id not in speaker_info:
                        speaker_info[speaker1_id] = {
                            'dialect': dialects[0] if len(dialects) > 0 else 'UNKNOWN',
                            'filename': filename
                        }
                    
                    if speaker2_id not in speaker_info:
                        speaker_info[speaker2_id] = {
                            'dialect': dialects[1] if len(dialects) > 1 else dialects[0],
                            'filename': filename
                        }
                except:
                    pass
    
    print(f"✅ 解析完成: {len(speaker_info)}个说话人")
    return speaker_info


def create_training_manifest(
    audio_files: list,
    output_json: str,
    speaker_info: dict = None,
    use_whisper_format: bool = True
):
    """
    创建训练数据清单（jsonl格式）
    
    注意：SeniorTalk数据可能没有单独的转录文本文件，
    需要先用Whisper-large-v3生成伪标签
    """
    print(f"📝 创建训练清单: {output_json}")
    
    manifest_data = []
    
    for audio_path in tqdm(audio_files, desc="处理音频"):
        audio_path = Path(audio_path)
        
        # 从文件名提取信息
        # 格式: Elderly0122S0001W0026.wav
        filename = audio_path.name
        
        # 提取speaker ID (S0001)
        try:
            speaker_id = filename.split('S')[1].split('W')[0]
            speaker_id = f"S{speaker_id}"
        except:
            speaker_id = "unknown"
        
        # 获取方言信息
        dialect = "unknown"
        if speaker_info and speaker_id in speaker_info:
            dialect = speaker_info[speaker_id].get('dialect', 'unknown')
        
        # 检查音频时长
        try:
            y, sr = librosa.load(str(audio_path), sr=16000)
            duration = len(y) / sr
            
            # 过滤太短或太长的音频
            if duration < 0.5 or duration > 30:
                continue
            
        except Exception as e:
            print(f"⚠️ 跳过损坏的音频: {audio_path} ({e})")
            continue
        
        item = {
            "audio_path": str(audio_path),
            "text": "",  # 需要后续添加转录文本
            "speaker_id": speaker_id,
            "dialect": dialect,
            "duration": duration,
            "source": "SeniorTalk"
        }
        
        manifest_data.append(item)
    
    # 保存为jsonl
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in manifest_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 创建清单完成: {len(manifest_data)}条记录")
    print(f"💾 保存到: {output_json}")
    
    return manifest_data


def generate_pseudo_labels(manifest_path: str, model_name: str = "openai/whisper-large-v3"):
    """
    使用Whisper-large-v3为SeniorTalk数据生成伪标签
    """
    print(f"🤖 开始生成伪标签（使用{model_name}）")
    print("⏳ 这可能需要较长时间...")
    
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    
    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎮 使用设备: {device}")
    
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    # 读取清单
    manifest_data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            manifest_data.append(json.loads(line))
    
    # 生成转录
    updated_data = []
    
    for item in tqdm(manifest_data, desc="生成转录"):
        audio_path = item['audio_path']
        
        try:
            # 加载音频
            audio_array, sr = librosa.load(audio_path, sr=16000)
            
            # 处理音频
            input_features = processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(device)
            
            # 生成转录
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    language="zh",
                    task="transcribe"
                )
            
            transcription = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            item['text'] = transcription.strip()
            updated_data.append(item)
            
        except Exception as e:
            print(f"⚠️ 跳过: {audio_path} ({e})")
            continue
    
    # 保存更新的清单
    output_path = manifest_path.replace('.jsonl', '_labeled.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in updated_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 伪标签生成完成: {len(updated_data)}条")
    print(f"💾 保存到: {output_path}")
    
    return output_path


def split_dataset(manifest_path: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    划分训练集、验证集、测试集
    """
    print(f"📊 划分数据集...")
    
    # 读取数据
    data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # 随机打乱
    import random
    random.seed(42)
    random.shuffle(data)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # 保存
    base_path = Path(manifest_path).parent
    
    train_path = base_path / "seniortalk_train.jsonl"
    val_path = base_path / "seniortalk_val.jsonl"
    test_path = base_path / "seniortalk_test.jsonl"
    
    for data_subset, path in [(train_data, train_path), (val_data, val_path), (test_data, test_path)]:
        with open(path, 'w', encoding='utf-8') as f:
            for item in data_subset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 数据划分完成:")
    print(f"   训练集: {len(train_data)}条 ({train_path})")
    print(f"   验证集: {len(val_data)}条 ({val_path})")
    print(f"   测试集: {len(test_data)}条 ({test_path})")
    
    return train_path, val_path, test_path


def main():
    # 配置路径
    TAR_PATH = "/data/AD_predict/data/raw/audio/seniortalk_asr_single/sentence_data/wav/train/train-0001.tar"
    UTTERANCE_INFO = "/data/AD_predict/data/raw/audio/seniortalk_asr_single/sentence_data/UTTERANCEINFO.txt"
    OUTPUT_DIR = "/data/AD_predict/data/seniortalk_processed"
    
    print("="*80)
    print("🚀 SeniorTalk数据整合流程")
    print("="*80)
    
    # 询问用户提取多少样本
    print("\n当前发现5171个音频文件")
    print("建议:")
    print("  - 快速测试: 500条")
    print("  - 中等规模: 2000条")
    print("  - 完整数据: 全部5171条")
    
    choice = input("\n请选择 [1=500条, 2=2000条, 3=全部, 或输入自定义数量]: ").strip()
    
    if choice == "1":
        max_samples = 500
    elif choice == "2":
        max_samples = 2000
    elif choice == "3":
        max_samples = None
    else:
        try:
            max_samples = int(choice)
        except:
            max_samples = 500
            print(f"使用默认值: {max_samples}条")
    
    # 步骤1: 解压音频
    print("\n" + "="*80)
    print("步骤1: 解压音频文件")
    print("="*80)
    audio_files = extract_seniortalk_tar(TAR_PATH, OUTPUT_DIR, max_samples)
    
    # 步骤2: 解析说话人信息
    print("\n" + "="*80)
    print("步骤2: 解析说话人信息")
    print("="*80)
    speaker_info = parse_utterance_info(UTTERANCE_INFO)
    
    # 步骤3: 创建清单
    print("\n" + "="*80)
    print("步骤3: 创建数据清单")
    print("="*80)
    manifest_path = os.path.join(OUTPUT_DIR, "seniortalk_manifest.jsonl")
    manifest_data = create_training_manifest(audio_files, manifest_path, speaker_info)
    
    # 步骤4: 生成伪标签（可选）
    print("\n" + "="*80)
    print("步骤4: 生成伪标签")
    print("="*80)
    print("⚠️ 注意: 这一步会很慢，如果SeniorTalk已经有转录文本，可以跳过")
    
    generate_labels = input("是否生成伪标签? [y/n]: ").strip().lower()
    
    if generate_labels == 'y':
        labeled_manifest = generate_pseudo_labels(manifest_path)
        manifest_path = labeled_manifest
    else:
        print("⏭️ 跳过伪标签生成")
        print("💡 请手动为清单添加转录文本")
    
    # 步骤5: 划分数据集
    print("\n" + "="*80)
    print("步骤5: 划分数据集")
    print("="*80)
    train_path, val_path, test_path = split_dataset(manifest_path)
    
    print("\n" + "="*80)
    print("✅ 完成！")
    print("="*80)
    print(f"\n下一步:")
    print(f"1. 查看数据: {OUTPUT_DIR}")
    print(f"2. 修改 scripts/1_prepare_dataset.py 整合SeniorTalk数据")
    print(f"3. 重新训练模型")


if __name__ == "__main__":
    main()

