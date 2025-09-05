#!/usr/bin/env python3
"""
下载SeniorTalk数据集的最小样本
只下载少量文件用于测试
"""
import os
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_minimal_seniortalk(output_dir="data/raw/audio/seniortalk_minimal"):
    """下载SeniorTalk的最小测试样本"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 下载目录: {output_path.absolute()}")
    logger.info("🔽 开始下载SeniorTalk最小样本...")
    
    # 设置镜像源
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    repo_id = "BAAI/SeniorTalk"
    
    # 需要下载的基础文件
    basic_files = [
        "README.md",
        "SPKINFO.txt", 
        "UTTERANCEINFO.txt",
        ".gitattributes"
    ]
    
    # 只下载几个转录文件作为样本
    sample_transcript_files = [
        "transcript/001&002-81&78-MF-BEIJING&SHANDONG-HUAWEI MGA-AL00-1.txt",
        "transcript/003&004-77&76-FF-HANGZHOU&HANGZHOU-HONOR AL10.txt",
        "transcript/005&006-76&77-FM-BEIJING&BEIJING-Iphone 13pro.txt"
    ]
    
    downloaded_files = []
    
    try:
        # 下载基础文件
        logger.info("📋 下载基础信息文件...")
        for file in basic_files:
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
        
        # 下载样本转录文件
        logger.info("📝 下载样本转录文件...")
        for file in sample_transcript_files:
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
        
        # 尝试下载一个小的音频文件 (如果存在的话)
        logger.info("🎵 尝试下载一个音频样本...")
        
        # 这些是可能的音频文件路径 (基于数据集结构猜测)
        possible_audio_files = [
            "wav/train/train-0001.tar",  # 如果是tar包
            "wav/test/test-0001.tar"     # 测试集可能较小
        ]
        
        for audio_file in possible_audio_files:
            try:
                # 首先检查文件是否存在
                from huggingface_hub import HfApi
                api = HfApi()
                
                # 尝试获取仓库信息
                repo_info = api.repo_info(repo_id, repo_type="dataset")
                
                logger.info(f"🔍 检查音频文件: {audio_file}")
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=audio_file,
                    repo_type="dataset",
                    local_dir=str(output_path),
                    local_dir_use_symlinks=False
                )
                logger.info(f"✅ 音频文件下载成功: {audio_file}")
                downloaded_files.append(local_path)
                break  # 只下载一个音频文件就够了
                
            except Exception as e:
                logger.warning(f"⚠️ 音频文件不可用 {audio_file}: {e}")
                continue
        
        # 创建数据集清单
        manifest = {
            "dataset_name": "SeniorTalk Minimal Sample",
            "source": repo_id,
            "download_path": str(output_path.absolute()),
            "description": "SeniorTalk数据集的最小测试样本",
            "downloaded_files": [str(Path(f).relative_to(output_path)) for f in downloaded_files],
            "total_files": len(downloaded_files),
            "note": "这是一个最小测试样本，包含基础信息和少量转录文件"
        }
        
        manifest_file = output_path / "download_manifest.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n🎉 下载完成!")
        logger.info(f"📁 文件位置: {output_path.absolute()}")
        logger.info(f"📊 总共下载: {len(downloaded_files)} 个文件")
        logger.info(f"📋 详细清单: {manifest_file}")
        
        return str(output_path.absolute())
        
    except Exception as e:
        logger.error(f"❌ 下载过程出现错误: {e}")
        return None

def list_downloaded_files(download_path="data/raw/audio/seniortalk_minimal"):
    """列出已下载的文件"""
    
    path = Path(download_path)
    if not path.exists():
        print(f"❌ 目录不存在: {path}")
        return
    
    print(f"\n📂 SeniorTalk 最小样本文件列表:")
    print(f"位置: {path.absolute()}")
    print("-" * 50)
    
    for item in sorted(path.rglob("*")):
        if item.is_file():
            size = item.stat().st_size
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024*1024:
                size_str = f"{size/1024:.1f}KB"
            else:
                size_str = f"{size/(1024*1024):.1f}MB"
            
            rel_path = item.relative_to(path)
            print(f"📄 {rel_path} ({size_str})")

def main():
    print("🚀 SeniorTalk 最小样本下载工具")
    print("=" * 50)
    print("只下载少量文件用于测试，节省空间和时间")
    print()
    
    # 下载最小样本
    result = download_minimal_seniortalk()
    
    if result:
        print(f"\n✅ 下载成功!")
        print(f"📁 数据位置: {result}")
        
        # 列出下载的文件
        list_downloaded_files(result)
        
        print(f"\n💡 使用方法:")
        print(f"1. 查看 README.md 了解数据集结构")
        print(f"2. 查看 transcript/ 目录中的转录文件")
        print(f"3. 查看 download_manifest.json 了解下载详情")
        
    else:
        print("❌ 下载失败")

if __name__ == "__main__":
    main() 