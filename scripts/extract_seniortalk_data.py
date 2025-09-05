#!/usr/bin/env python3
"""
解压SeniorTalk数据集并整理为可用的格式
"""
import os
import tarfile
import logging
from pathlib import Path
import json
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_tar_with_limit(tar_path: str, extract_dir: str, max_files: int = 5):
    """解压tar文件，限制文件数量用于测试"""
    
    tar_path = Path(tar_path)
    extract_dir = Path(extract_dir)
    
    logger.info(f"📦 开始解压: {tar_path.name}")
    logger.info(f"📁 解压到: {extract_dir}")
    logger.info(f"🔢 限制文件数量: {max_files}")
    
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_files = []
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()
            
            # 过滤出音频文件
            audio_members = [m for m in members if m.isfile() and m.name.endswith('.wav')]
            
            logger.info(f"📊 tar包包含 {len(audio_members)} 个wav文件")
            
            # 只提取指定数量的文件
            selected_members = audio_members[:max_files]
            
            for i, member in enumerate(selected_members):
                logger.info(f"⏳ 解压 {i+1}/{len(selected_members)}: {member.name}")
                
                # 解压文件
                tar.extract(member, extract_dir)
                
                # 记录文件信息
                extracted_path = extract_dir / member.name
                if extracted_path.exists():
                    file_size = extracted_path.stat().st_size / (1024 * 1024)  # MB
                    logger.info(f"  ✅ 大小: {file_size:.1f}MB")
                    
                    extracted_files.append({
                        'original_path': member.name,
                        'extracted_path': str(extracted_path),
                        'size_mb': file_size
                    })
                else:
                    logger.warning(f"  ❌ 解压失败: {member.name}")
            
            logger.info(f"🎉 成功解压 {len(extracted_files)} 个文件")
            return extracted_files
            
    except Exception as e:
        logger.error(f"❌ 解压失败: {e}")
        return []

def find_corresponding_transcripts(audio_files: list, base_dir: str):
    """查找对应的转录文件"""
    
    base_path = Path(base_dir)
    transcripts = {}
    
    logger.info("🔍 查找对应的转录文件...")
    
    # 查找可能的转录文件目录
    possible_transcript_dirs = [
        base_path / "sentence_data" / "transcript",
        base_path / "transcript", 
        base_path.parent / "transcript",
        base_path.parent / "sentence_data" / "transcript"
    ]
    
    transcript_files = []
    for transcript_dir in possible_transcript_dirs:
        if transcript_dir.exists():
            transcript_files.extend(list(transcript_dir.glob("*.txt")))
            logger.info(f"📝 在 {transcript_dir} 找到 {len(list(transcript_dir.glob('*.txt')))} 个txt文件")
    
    # 尝试匹配音频文件和转录文件
    for audio_info in audio_files:
        audio_path = Path(audio_info['extracted_path'])
        audio_stem = audio_path.stem
        
        # 尝试找到对应的转录文件
        matching_transcript = None
        for transcript_file in transcript_files:
            if audio_stem in transcript_file.stem or transcript_file.stem in audio_stem:
                matching_transcript = str(transcript_file)
                break
        
        transcripts[audio_stem] = matching_transcript
        
        if matching_transcript:
            logger.info(f"✅ 找到匹配: {audio_path.name} -> {Path(matching_transcript).name}")
        else:
            logger.warning(f"⚠️ 未找到转录: {audio_path.name}")
    
    return transcripts

def create_dataset_structure(extracted_files: list, transcripts: dict, output_dir: str):
    """创建标准的数据集结构"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🏗️ 创建数据集结构: {output_path}")
    
    # 创建标准目录结构
    audio_dir = output_path / "audio"
    text_dir = output_path / "text"
    metadata_dir = output_path / "metadata"
    
    for dir_path in [audio_dir, text_dir, metadata_dir]:
        dir_path.mkdir(exist_ok=True)
    
    dataset_items = []
    
    for i, audio_info in enumerate(extracted_files):
        audio_src = Path(audio_info['extracted_path'])
        audio_stem = audio_src.stem
        
        # 复制音频文件到标准位置
        audio_dest = audio_dir / f"sample_{i+1:03d}.wav"
        shutil.copy2(audio_src, audio_dest)
        
        # 处理转录文件
        transcript_text = ""
        transcript_src = transcripts.get(audio_stem)
        
        if transcript_src and Path(transcript_src).exists():
            try:
                with open(transcript_src, 'r', encoding='utf-8') as f:
                    transcript_text = f.read().strip()
                    
                # 保存转录文件
                text_dest = text_dir / f"sample_{i+1:03d}.txt"
                with open(text_dest, 'w', encoding='utf-8') as f:
                    f.write(transcript_text)
                    
                logger.info(f"📝 转录文本: {transcript_text[:50]}...")
                
            except Exception as e:
                logger.warning(f"⚠️ 读取转录失败 {transcript_src}: {e}")
                
                # 创建空转录文件
                text_dest = text_dir / f"sample_{i+1:03d}.txt"
                with open(text_dest, 'w', encoding='utf-8') as f:
                    f.write("# 转录文件读取失败")
        else:
            # 创建占位符转录文件
            text_dest = text_dir / f"sample_{i+1:03d}.txt"
            with open(text_dest, 'w', encoding='utf-8') as f:
                f.write(f"# 老年人语音样本 {i+1}")
            transcript_text = f"老年人语音样本 {i+1}"
        
        # 记录数据项
        dataset_item = {
            'id': f"sample_{i+1:03d}",
            'audio_file': str(audio_dest.relative_to(output_path)),
            'text_file': str(text_dest.relative_to(output_path)),
            'transcript': transcript_text,
            'original_audio_path': audio_info['original_path'],
            'size_mb': audio_info['size_mb']
        }
        
        dataset_items.append(dataset_item)
        logger.info(f"✅ 处理完成: {dataset_item['id']}")
    
    # 创建数据集元数据
    metadata = {
        'dataset_name': 'SeniorTalk Sample Dataset',
        'source': 'BAAI/SeniorTalk',
        'description': '从SeniorTalk数据集提取的老年人语音样本',
        'total_samples': len(dataset_items),
        'structure': {
            'audio/': '音频文件 (.wav)',
            'text/': '转录文件 (.txt)',
            'metadata/': '元数据文件'
        },
        'samples': dataset_items
    }
    
    # 保存元数据
    metadata_file = metadata_dir / "dataset_info.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # 创建简单的索引文件
    index_file = output_path / "index.txt"
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write("# SeniorTalk Sample Dataset Index\n")
        f.write("# Format: audio_file|text_file|transcript\n\n")
        
        for item in dataset_items:
            f.write(f"{item['audio_file']}|{item['text_file']}|{item['transcript']}\n")
    
    logger.info(f"📋 数据集创建完成: {output_path}")
    logger.info(f"📊 包含 {len(dataset_items)} 个样本")
    logger.info(f"📋 元数据文件: {metadata_file}")
    logger.info(f"📄 索引文件: {index_file}")
    
    return metadata

def main():
    """主函数"""
    
    print("🚀 SeniorTalk数据解压与整理工具")
    print("=" * 60)
    
    # 查找tar文件
    tar_file = "data/raw/audio/seniortalk_asr_single/sentence_data/wav/train/train-0001.tar"
    
    if not Path(tar_file).exists():
        logger.error(f"❌ 未找到tar文件: {tar_file}")
        return
    
    # 设置路径
    extract_base_dir = "data/processed/seniortalk_extracted"
    final_dataset_dir = "data/processed/seniortalk_samples"
    
    logger.info(f"📦 找到tar文件: {tar_file}")
    logger.info(f"📁 解压到: {extract_base_dir}")
    logger.info(f"🎯 最终数据集: {final_dataset_dir}")
    
    # 第1步: 解压tar文件（限制数量）
    extracted_files = extract_tar_with_limit(
        tar_path=tar_file,
        extract_dir=extract_base_dir,
        max_files=5  # 只解压5个文件用于测试
    )
    
    if not extracted_files:
        logger.error("❌ 没有成功解压任何文件")
        return
    
    # 第2步: 查找转录文件
    transcripts = find_corresponding_transcripts(
        audio_files=extracted_files,
        base_dir=Path(tar_file).parent.parent.parent.parent  # sentence_data目录
    )
    
    # 第3步: 创建标准数据集结构
    metadata = create_dataset_structure(
        extracted_files=extracted_files,
        transcripts=transcripts,
        output_dir=final_dataset_dir
    )
    
    # 显示结果
    print(f"\n🎉 数据整理完成!")
    print(f"📁 数据集位置: {final_dataset_dir}")
    print(f"📊 样本数量: {metadata['total_samples']}")
    
    print(f"\n📂 目录结构:")
    print(f"  {final_dataset_dir}/")
    print(f"  ├── audio/          # 音频文件")
    print(f"  ├── text/           # 转录文件")
    print(f"  ├── metadata/       # 元数据")
    print(f"  └── index.txt       # 索引文件")
    
    print(f"\n💡 接下来:")
    print(f"  1. 检查数据: ls -la {final_dataset_dir}/audio/")
    print(f"  2. 修改 test_enhanced_asr.py 使用新数据")
    print(f"  3. 运行测试: python scripts/test_enhanced_asr.py")

if __name__ == "__main__":
    main() 