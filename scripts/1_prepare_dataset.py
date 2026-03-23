"""
数据预处理脚本
功能：
1. 读取Excel文件，解析方言标签
2. 匹配音频文件和转录文本
3. 构建带方言提示的Whisper输入格式
4. 对样本少的类别进行数据增强
5. 分割并保存训练集和验证集
"""

import os
import sys
import pandas as pd
import json
import yaml
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, Audio
from tqdm import tqdm
import shutil

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.audio_augment import AudioAugmentor


def load_config(config_path="configs/training_args.yaml"):
    """加载配置文件"""
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    config_full_path = project_root / config_path
    
    with open(config_full_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_excel_data(excel_path):
    """
    加载Excel文件，获取方言标签信息
    
    Returns:
        DataFrame with columns: up主, 视频名称, url, dialect_label
    """
    df = pd.read_excel(excel_path)
    print(f"从Excel加载了 {len(df)} 行数据")
    print(f"方言分布:\n{df['dialect_label'].value_counts()}")
    return df


def find_audio_and_transcript_pairs(base_dir):
    """
    查找音频文件和对应的转录文本
    
    Returns:
        list of dicts: [{"audio_path": path, "transcript": text, "audio_id": id}, ...]
    """
    pairs = []
    
    # 查找elderly_audios目录下的音频文件
    audio_dir = os.path.join(base_dir, "elderly_audios")
    result_dir = os.path.join(base_dir, "result")
    
    if os.path.exists(audio_dir):
        audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
        
        for audio_file in audio_files:
            audio_path = os.path.join(audio_dir, audio_file)
            
            # 提取音频ID (例如: elderly_audio_0001.wav -> 0001)
            audio_id = audio_file.replace('elderly_audio_', '').replace('.wav', '')
            
            # 尝试找到对应的转录文本
            # 假设转录文本文件名类似 test1.txt, test100.txt 等
            # 需要根据实际情况调整映射逻辑
            transcript_path = os.path.join(result_dir, f"test{int(audio_id)}.txt")
            
            if os.path.exists(transcript_path):
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
                    # 将多行文本合并为单行
                    transcript = ' '.join(transcript.split('\n'))
                
                pairs.append({
                    "audio_path": audio_path,
                    "transcript": transcript,
                    "audio_id": audio_id
                })
    
    print(f"找到 {len(pairs)} 对音频-文本数据")
    return pairs


def map_dialect_labels(audio_transcript_pairs, excel_df):
    """
    将方言标签映射到音频-文本对
    
    由于Excel中的数据可能与音频文件不是一一对应，这里提供一个简化的映射策略：
    - 如果能通过视频名称匹配，则使用对应的方言标签
    - 否则，可以手动指定默认标签或跳过
    """
    labeled_data = []
    
    # 简化版本：如果Excel行数与音频文件数量匹配，则顺序对应
    # 实际使用时可能需要更复杂的匹配逻辑
    for i, pair in enumerate(audio_transcript_pairs):
        if i < len(excel_df):
            dialect = excel_df.iloc[i]['dialect_label']
            if pd.notna(dialect):  # 过滤掉NaN值
                pair['dialect_label'] = dialect
                labeled_data.append(pair)
    
    print(f"成功标注了 {len(labeled_data)} 条数据")
    
    # 打印方言分布
    dialect_counts = Counter([d['dialect_label'] for d in labeled_data])
    print(f"标注后的方言分布:\n{dict(dialect_counts)}")
    
    return labeled_data


def create_whisper_prompt_text(transcript, dialect_label):
    """
    创建带方言提示的Whisper输入文本
    
    格式: <|startoftranscript|><|zh|><|transcribe|><|notimestamps|><|dialect:LABEL|> TEXT
    注意：Whisper的特殊token需要在训练时正确处理
    """
    # 构建方言提示token
    dialect_token = f"<|dialect:{dialect_label}|>"
    
    # Whisper的标准格式
    # 在实际训练时，这些特殊token会被tokenizer正确处理
    prompt_text = f"<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>{dialect_token} {transcript}"
    
    return prompt_text


def augment_minority_dialects(labeled_data, augmentor, audio_base_dir, min_samples=10, aug_factor=4):
    """
    对样本数量少的方言类别进行数据增强
    
    Args:
        labeled_data: 原始数据列表
        augmentor: 音频增强器
        audio_base_dir: 音频基础目录
        min_samples: 少于此数量的类别将被增强
        aug_factor: 增强倍数
        
    Returns:
        augmented_data: 增强后的数据列表
    """
    # 统计各方言的样本数
    dialect_counts = Counter([d['dialect_label'] for d in labeled_data])
    minority_dialects = [dialect for dialect, count in dialect_counts.items() 
                        if count < min_samples]
    
    print(f"\n需要增强的方言类别 (样本数 < {min_samples}): {minority_dialects}")
    
    augmented_data = labeled_data.copy()
    
    # ⚠️ 修复：将增强音频保存到永久目录，而不是临时目录
    # 在elderly_audios同级创建augmented_audios目录
    aug_dir = os.path.join(audio_base_dir, "elderly_audios_augmented")
    os.makedirs(aug_dir, exist_ok=True)
    print(f"增强音频将保存到: {aug_dir}")
    
    # 统计增强过程
    successful_augmentations = 0
    failed_augmentations = 0
    created_files = []
    
    for data_item in tqdm(labeled_data, desc="数据增强"):
        if data_item['dialect_label'] in minority_dialects:
            # 对这个样本进行增强
            audio_path = data_item['audio_path']
            audio_id = data_item['audio_id']
            
            try:
                # 生成增强版本
                augmented_audios = augmentor.augment_audio_file(
                    audio_path, 
                    num_augmentations=aug_factor
                )
                
                # 保存增强音频并创建新的数据项
                for aug_idx, (audio_array, sr) in enumerate(augmented_audios):
                    aug_filename = f"aug_{audio_id}_{aug_idx}.wav"
                    aug_audio_path = os.path.join(aug_dir, aug_filename)
                    
                    # 保存增强音频到永久目录
                    import soundfile as sf
                    sf.write(aug_audio_path, audio_array, sr)
                    
                    # 验证文件是否成功创建
                    if os.path.exists(aug_audio_path) and os.path.getsize(aug_audio_path) > 0:
                        created_files.append(aug_audio_path)
                        successful_augmentations += 1
                        
                        # 创建新的数据项（复制原始项的信息）
                        aug_item = data_item.copy()
                        aug_item['audio_path'] = aug_audio_path
                        aug_item['audio_id'] = f"{audio_id}_aug_{aug_idx}"
                        aug_item['is_augmented'] = True
                        
                        augmented_data.append(aug_item)
                    else:
                        print(f"⚠️ 警告：增强音频文件创建失败: {aug_audio_path}")
                        failed_augmentations += 1
            
            except Exception as e:
                print(f"❌ 增强失败 {audio_path}: {e}")
                failed_augmentations += 1
                continue
    
    # 详细报告增强结果
    print(f"\n" + "="*60)
    print("📊 数据增强结果报告")
    print("="*60)
    print(f"✅ 成功增强样本数: {successful_augmentations}")
    print(f"❌ 失败增强样本数: {failed_augmentations}")
    print(f"📁 增强音频目录: {aug_dir}")
    print(f"📁 实际创建文件数: {len(created_files)}")
    print(f"📈 增强后总样本数: {len(augmented_data)} (原始: {len(labeled_data)})")
    
    # 验证增强音频文件的存在性
    print(f"\n🔍 增强音频文件验证:")
    existing_files = 0
    missing_files = 0
    
    for aug_item in augmented_data:
        if aug_item.get('is_augmented', False):
            audio_path = aug_item['audio_path']
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                existing_files += 1
            else:
                missing_files += 1
                print(f"⚠️ 缺失文件: {audio_path}")
    
    print(f"✅ 存在的增强音频文件: {existing_files}")
    print(f"❌ 缺失的增强音频文件: {missing_files}")
    
    # 打印增强后的方言分布
    final_dialect_counts = Counter([d['dialect_label'] for d in augmented_data])
    print(f"\n📊 增强后的方言分布:")
    for dialect, count in final_dialect_counts.items():
        original_count = dialect_counts.get(dialect, 0)
        print(f"  {dialect}: {original_count} → {count} (+{count-original_count})")
    
    return augmented_data


def check_existing_augmented_audio(audio_base_dir):
    """
    检查已存在的增强音频文件
    
    Args:
        audio_base_dir: 音频基础目录
        
    Returns:
        dict: 增强音频文件统计信息
    """
    aug_dir = os.path.join(audio_base_dir, "elderly_audios_augmented")
    
    if not os.path.exists(aug_dir):
        print(f"📁 增强音频目录不存在: {aug_dir}")
        return {"exists": False, "count": 0, "files": []}
    
    # 获取所有增强音频文件
    aug_files = [f for f in os.listdir(aug_dir) if f.endswith('.wav') and f.startswith('aug_')]
    
    print(f"\n🔍 检查现有增强音频文件:")
    print(f"📁 目录: {aug_dir}")
    print(f"📊 文件数量: {len(aug_files)}")
    
    if aug_files:
        print(f"📋 前10个文件示例:")
        for i, file in enumerate(sorted(aug_files)[:10]):
            file_path = os.path.join(aug_dir, file)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            print(f"  {i+1}. {file} ({file_size/1024/1024:.1f}MB)")
        
        if len(aug_files) > 10:
            print(f"  ... 还有 {len(aug_files)-10} 个文件")
    
    return {
        "exists": True,
        "count": len(aug_files),
        "files": aug_files,
        "directory": aug_dir
    }


def prepare_dataset_for_whisper(data_list):
    """
    准备Hugging Face Dataset格式的数据
    
    Returns:
        Dataset with columns: audio, text, dialect_label
    """
    dataset_dict = {
        "audio": [],
        "text": [],
        "dialect_label": []
    }
    
    for item in data_list:
        # Whisper需要的文本格式（包含方言提示）
        prompt_text = create_whisper_prompt_text(
            item['transcript'], 
            item['dialect_label']
        )
        
        dataset_dict["audio"].append(item['audio_path'])
        dataset_dict["text"].append(prompt_text)
        dataset_dict["dialect_label"].append(item['dialect_label'])
    
    # 创建Dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    # 将audio列转换为Audio特征（会自动加载音频）
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    return dataset


def main():
    """主函数"""
    print("=" * 80)
    print("开始数据预处理")
    print("=" * 80)
    
    # 1. 加载配置
    config = load_config()
    excel_path = config['data']['excel_path']
    audio_base_dir = config['data']['audio_base_dir'].replace('/elderly_audios', '')
    processed_data_dir = config['data']['processed_data_dir']
    train_split_ratio = config['data']['train_split_ratio']
    
    # 数据增强配置
    enable_augmentation = config['data']['augmentation']['enable']
    min_samples_threshold = config['data']['augmentation']['min_samples_threshold']
    aug_factor = config['data']['augmentation']['augmentation_factor']
    
    # 2. 加载Excel数据（包含方言标签）
    print("\n步骤 1: 加载Excel数据...")
    excel_df = load_excel_data(excel_path)
    
    # 3. 查找音频和转录文本对
    print("\n步骤 2: 查找音频和转录文本...")
    audio_transcript_pairs = find_audio_and_transcript_pairs(audio_base_dir)
    
    if len(audio_transcript_pairs) == 0:
        print("错误: 没有找到音频-文本对！")
        print("请检查音频文件路径和转录文本路径是否正确。")
        return
    
    # 4. 映射方言标签
    print("\n步骤 3: 映射方言标签...")
    labeled_data = map_dialect_labels(audio_transcript_pairs, excel_df)
    
    if len(labeled_data) == 0:
        print("错误: 没有成功标注任何数据！")
        return
    
    # 5. 先分割训练集和验证集（在原始数据上分割）
    print(f"\n步骤 4: 分割数据集 (训练集比例={train_split_ratio})...")
    print("⚠️  重要：在原始数据上分割，确保验证集不包含增强音频")
    
    # 检查是否可以使用分层采样（每个类别至少要有2个样本）
    from collections import Counter
    dialect_counts = Counter([d['dialect_label'] for d in labeled_data])
    min_samples = min(dialect_counts.values())
    
    if min_samples >= 2:
        # 可以使用分层采样
        print("✓ 使用分层采样（每个方言类别至少有2个样本）")
        train_data, val_data = train_test_split(
            labeled_data, 
            train_size=train_split_ratio,
            random_state=42,
            stratify=[d['dialect_label'] for d in labeled_data]
        )
    else:
        # 不使用分层采样
        print(f"⚠️  某些方言类别样本数<2，使用随机采样")
        train_data, val_data = train_test_split(
            labeled_data, 
            train_size=train_split_ratio,
            random_state=42
        )
    
    print(f"原始训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(val_data)} (仅原始音频)")
    
    # 6. 检查现有增强音频
    print(f"\n步骤 5: 检查现有增强音频...")
    existing_aug_info = check_existing_augmented_audio(audio_base_dir)
    
    # 7. 数据增强（仅对训练集）
    if enable_augmentation:
        print(f"\n步骤 6: 数据增强 (仅训练集, 阈值={min_samples_threshold}, 倍数={aug_factor})...")
        
        # 如果已有增强音频，询问是否重新生成
        if existing_aug_info["exists"] and existing_aug_info["count"] > 0:
            print(f"⚠️ 发现已存在 {existing_aug_info['count']} 个增强音频文件")
            print(f"📁 位置: {existing_aug_info['directory']}")
            print("🔄 将重新生成增强音频（覆盖现有文件）")
        
        augmentor = AudioAugmentor(sample_rate=16000)
        train_data = augment_minority_dialects(
            train_data,  # 只增强训练集
            augmentor,
            audio_base_dir,  # 传入音频基础目录
            min_samples=min_samples_threshold,
            aug_factor=aug_factor
        )
        print(f"✓ 增强后训练集样本数: {len(train_data)}")
    else:
        print("\n步骤 6: 跳过数据增强")
    
    print(f"\n最终数据集统计:")
    print(f"  训练集: {len(train_data)} (包含增强音频)")
    print(f"  验证集: {len(val_data)} (仅原始音频)")
    
    # 8. 转换为Hugging Face Dataset格式
    print("\n步骤 7: 创建Hugging Face Dataset...")
    train_dataset = prepare_dataset_for_whisper(train_data)
    val_dataset = prepare_dataset_for_whisper(val_data)
    
    # 9. 保存数据集
    print(f"\n步骤 8: 保存数据集到 {processed_data_dir}...")
    os.makedirs(processed_data_dir, exist_ok=True)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    dataset_dict.save_to_disk(processed_data_dir)
    
    print("\n" + "=" * 80)
    print("数据预处理完成！")
    print(f"处理后的数据已保存到: {processed_data_dir}")
    print("=" * 80)
    
    # 打印一些示例数据（使用原始数据，避免触发音频解码）
    print("\n示例数据 (前3条):")
    for i in range(min(3, len(train_data))):
        print(f"\n样本 {i+1}:")
        print(f"  方言: {train_data[i]['dialect_label']}")
        print(f"  转录: {train_data[i]['transcript'][:100]}...")
        print(f"  音频: {train_data[i]['audio_path']}")
    
    # 保存数据统计信息
    stats = {
        "total_samples": len(labeled_data),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "dialect_distribution": dict(Counter([d['dialect_label'] for d in labeled_data])),
        "augmentation_enabled": enable_augmentation
    }
    
    stats_path = os.path.join(processed_data_dir, "dataset_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n数据集统计信息已保存到: {stats_path}")


if __name__ == "__main__":
    main()

