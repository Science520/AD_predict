#!/usr/bin/env python3
"""
下载SeniorTalk ASR数据集的单个tar包
从sentence_data目录下载一个tar文件用于测试
"""
import os
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_single_asr_tar(output_dir="data/raw/audio/seniortalk_asr_single"):
    """下载ASR数据集的单个tar包"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 下载目录: {output_path.absolute()}")
    logger.info("🔽 开始下载SeniorTalk ASR单个tar包...")
    
    # 设置镜像源
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    repo_id = "BAAI/SeniorTalk"
    
    # ASR数据集需要的基础文件
    asr_basic_files = [
        "sentence_data/SPKINFO.txt",
        "sentence_data/UTTERANCEINFO.txt"
    ]
    
    # 可能的tar文件（按优先级排序，测试集通常较小）
    possible_tar_files = [
        "sentence_data/wav/test/test-0001.tar",
        "sentence_data/wav/dev/dev-0001.tar", 
        "sentence_data/wav/train/train-0001.tar"
    ]
    
    downloaded_files = []
    
    try:
        # 首先下载README
        logger.info("📋 下载README文件...")
        try:
            readme_path = hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                repo_type="dataset",
                local_dir=str(output_path),
                local_dir_use_symlinks=False
            )
            logger.info(f"✅ README.md")
            downloaded_files.append(readme_path)
        except Exception as e:
            logger.warning(f"⚠️ README下载失败: {e}")
        
        # 下载ASR基础信息文件
        logger.info("📊 下载ASR基础信息文件...")
        for file in asr_basic_files:
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    repo_type="dataset",
                    local_dir=str(output_path),
                    local_dir_use_symlinks=False
                )
                logger.info(f"✅ {file}")
                downloaded_files.append(local_path)
            except Exception as e:
                logger.warning(f"⚠️ 跳过 {file}: {e}")
        
        # 尝试下载一个tar文件
        logger.info("🎵 尝试下载一个音频tar包...")
        tar_downloaded = False
        
        for tar_file in possible_tar_files:
            try:
                logger.info(f"🔍 尝试下载: {tar_file}")
                
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=tar_file,
                    repo_type="dataset",
                    local_dir=str(output_path),
                    local_dir_use_symlinks=False
                )
                
                # 获取文件大小
                file_size = Path(local_path).stat().st_size
                size_mb = file_size / (1024 * 1024)
                
                logger.info(f"✅ 成功下载: {tar_file} ({size_mb:.1f}MB)")
                downloaded_files.append(local_path)
                tar_downloaded = True
                break
                
            except Exception as e:
                logger.warning(f"⚠️ {tar_file} 不可用: {e}")
                continue
        
        if not tar_downloaded:
            logger.warning("❌ 没有成功下载任何tar文件")
        
        # 尝试下载一些转录文件样本
        logger.info("📝 尝试下载一些转录文件样本...")
        transcript_samples = [
            "sentence_data/transcript/sample_001.txt",
            "sentence_data/transcript/sample_002.txt"
        ]
        
        for transcript_file in transcript_samples:
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=transcript_file,
                    repo_type="dataset",
                    local_dir=str(output_path),
                    local_dir_use_symlinks=False
                )
                logger.info(f"✅ {transcript_file}")
                downloaded_files.append(local_path)
            except Exception as e:
                logger.warning(f"⚠️ 跳过转录文件 {transcript_file}: {e}")
        
        # 创建下载清单
        manifest = {
            "dataset_name": "SeniorTalk ASR Single Tar",
            "source": repo_id,
            "download_path": str(output_path.absolute()),
            "description": "SeniorTalk ASR数据集的单个tar包测试样本",
            "downloaded_files": [str(Path(f).relative_to(output_path)) for f in downloaded_files],
            "total_files": len(downloaded_files),
            "tar_file_downloaded": tar_downloaded,
            "note": "包含ASR数据集的一个tar音频包和相关配置文件"
        }
        
        manifest_file = output_path / "download_manifest.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n🎉 下载完成!")
        logger.info(f"📁 文件位置: {output_path.absolute()}")
        logger.info(f"📊 总共下载: {len(downloaded_files)} 个文件")
        logger.info(f"🎵 包含tar包: {'是' if tar_downloaded else '否'}")
        logger.info(f"📋 详细清单: {manifest_file}")
        
        return str(output_path.absolute())
        
    except Exception as e:
        logger.error(f"❌ 下载过程出现错误: {e}")
        return None

def extract_tar_sample(download_path="data/raw/audio/seniortalk_asr_single"):
    """提取tar包中的一些样本文件"""
    
    import tarfile
    
    path = Path(download_path)
    tar_files = list(path.rglob("*.tar"))
    
    if not tar_files:
        logger.warning("❌ 未找到tar文件")
        return
    
    tar_file = tar_files[0]
    logger.info(f"🗂️ 提取tar文件样本: {tar_file.name}")
    
    extract_dir = path / "extracted_samples"
    extract_dir.mkdir(exist_ok=True)
    
    try:
        with tarfile.open(tar_file, 'r') as tar:
            members = tar.getmembers()
            wav_members = [m for m in members if m.isfile() and m.name.endswith('.wav')]
            
            # 只提取前3个wav文件作为样本
            sample_count = min(3, len(wav_members))
            
            logger.info(f"📁 tar包包含 {len(wav_members)} 个wav文件，提取前 {sample_count} 个")
            
            for i, member in enumerate(wav_members[:sample_count]):
                tar.extract(member, extract_dir)
                size = member.size / (1024 * 1024)
                logger.info(f"✅ 提取样本 {i+1}: {member.name} ({size:.2f}MB)")
            
            logger.info(f"🎉 样本提取完成到: {extract_dir}")
            
    except Exception as e:
        logger.error(f"❌ 提取失败: {e}")

def list_downloaded_files(download_path="data/raw/audio/seniortalk_asr_single"):
    """列出已下载的文件"""
    
    path = Path(download_path)
    if not path.exists():
        print(f"❌ 目录不存在: {path}")
        return
    
    print(f"\n📂 SeniorTalk ASR单个tar包文件列表:")
    print(f"位置: {path.absolute()}")
    print("-" * 60)
    
    total_size = 0
    for item in sorted(path.rglob("*")):
        if item.is_file():
            size = item.stat().st_size
            total_size += size
            
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024*1024:
                size_str = f"{size/1024:.1f}KB"
            else:
                size_str = f"{size/(1024*1024):.1f}MB"
            
            rel_path = item.relative_to(path)
            
            # 标记不同类型的文件
            if item.suffix == '.tar':
                icon = "🗂️"
            elif item.suffix == '.wav':
                icon = "🎵"
            elif item.suffix == '.txt':
                icon = "📝"
            elif item.suffix == '.json':
                icon = "📋"
            else:
                icon = "📄"
            
            print(f"{icon} {rel_path} ({size_str})")
    
    total_mb = total_size / (1024 * 1024)
    print(f"\n📊 总大小: {total_mb:.1f}MB")

def main():
    print("🚀 SeniorTalk ASR单个tar包下载工具")
    print("=" * 60)
    print("专门下载sentence_data目录下的一个tar包用于测试")
    print()
    
    # 下载单个tar包
    result = download_single_asr_tar()
    
    if result:
        print(f"\n✅ 下载成功!")
        print(f"📁 数据位置: {result}")
        
        # 列出下载的文件
        list_downloaded_files(result)
        
        # 询问是否提取样本
        print(f"\n❓ 是否提取tar包中的音频样本？")
        print(f"💡 这将提取几个wav文件到 extracted_samples/ 目录")
        print(f"🔧 运行以下命令来提取样本:")
        print(f"   python -c \"from scripts.download_asr_single_tar import extract_tar_sample; extract_tar_sample()\"")
        
    else:
        print("❌ 下载失败")

if __name__ == "__main__":
    main() 