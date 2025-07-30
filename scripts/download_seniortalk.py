#!/usr/bin/env python3
"""
SeniorTalk数据集下载脚本 - 改进版
"""
import os
import sys
import logging
from pathlib import Path
import argparse
import tarfile
import json

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_seniortalk(output_dir: str = "data/raw/audio"):
    """下载SeniorTalk数据集"""
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("开始下载SeniorTalk数据集...")
    
    try:
        # 尝试使用huggingface_hub直接下载
        from huggingface_hub import snapshot_download
        
        # 下载整个仓库
        repo_path = snapshot_download(
            repo_id="BAAI/SeniorTalk",
            repo_type="dataset",
            cache_dir=str(output_path / "cache"),
            local_dir=str(output_path / "seniortalk_raw")
        )
        
        logger.info(f"数据集下载到: {repo_path}")
        
        # 检查下载的文件
        seniortalk_path = Path(output_path / "seniortalk_raw")
        if seniortalk_path.exists():
            logger.info("下载成功! 文件结构:")
            for item in seniortalk_path.rglob("*"):
                if item.is_file():
                    logger.info(f"  {item.relative_to(seniortalk_path)}")
        
        # 创建简化的数据清单
        manifest_data = {
            "dataset_name": "SeniorTalk",
            "total_files": len(list(seniortalk_path.rglob("*.tar"))),
            "data_path": str(seniortalk_path),
            "description": "中文老年人对话数据集，包含75-85岁老人的自然对话录音"
        }
        
        manifest_path = output_path / "dataset_info.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据集信息保存到: {manifest_path}")
        
        return True
        
    except ImportError:
        logger.error("需要安装 huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"下载失败: {e}")
        logger.info("请确保已正确登录HuggingFace并有权访问该数据集")
        return False

def extract_sample_data(output_dir: str = "data/raw/audio"):
    """提取少量样本数据用于测试"""
    
    output_path = Path(output_dir)
    seniortalk_path = output_path / "seniortalk_raw"
    
    if not seniortalk_path.exists():
        logger.error("未找到SeniorTalk数据，请先运行下载")
        return False
    
    # 查找tar文件
    tar_files = list(seniortalk_path.rglob("*.tar"))
    if not tar_files:
        logger.error("未找到tar文件")
        return False
    
    # 创建样本目录
    sample_dir = output_path / "samples"
    sample_dir.mkdir(exist_ok=True)
    
    # 提取第一个tar文件的部分内容作为样本
    first_tar = tar_files[0]
    logger.info(f"正在从 {first_tar.name} 提取样本...")
    
    try:
        with tarfile.open(first_tar, 'r') as tar:
            members = tar.getmembers()[:5]  # 只提取前5个文件作为样本
            
            for member in members:
                if member.isfile() and member.name.endswith('.wav'):
                    tar.extract(member, sample_dir)
                    logger.info(f"提取样本: {member.name}")
        
        logger.info(f"样本文件提取到: {sample_dir}")
        return True
        
    except Exception as e:
        logger.error(f"提取样本失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="下载SeniorTalk数据集")
    parser.add_argument("--output_dir", type=str, default="data/raw/audio",
                       help="输出目录 (默认: data/raw/audio)")
    parser.add_argument("--extract_samples", action="store_true",
                       help="提取样本数据用于测试")
    
    args = parser.parse_args()
    
    if args.extract_samples:
        success = extract_sample_data(args.output_dir)
        if success:
            logger.info("✅ 样本数据提取完成!")
        else:
            logger.error("❌ 样本数据提取失败")
    else:
        success = download_seniortalk(args.output_dir)
        
        if success:
            logger.info("✅ SeniorTalk数据集下载完成!")
            logger.info("现在可以运行: python scripts/download_seniortalk.py --extract_samples 来提取样本")
            logger.info("或运行: python scripts/preprocess_data.py 来预处理数据")
        else:
            logger.error("❌ 数据集下载失败")
            sys.exit(1)

if __name__ == "__main__":
    main() 