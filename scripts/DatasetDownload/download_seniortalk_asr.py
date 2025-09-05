#!/usr/bin/env python3
"""
SeniorTalk ASR数据集下载脚本 - 专门用于下载少量测试样本
"""
import os
import sys
import logging
from pathlib import Path
import argparse
import tarfile
import json
import shutil

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_asr_samples(output_dir: str = "data/raw/audio", num_samples: int = 5):
    """下载SeniorTalk ASR数据集的少量测试样本"""
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"开始下载SeniorTalk ASR数据集的{num_samples}条测试样本...")
    
    try:
        # 尝试使用huggingface_hub直接下载
        from huggingface_hub import snapshot_download, hf_hub_download
        
        # 创建ASR数据目录
        asr_data_path = output_path / "sentence_data"
        asr_data_path.mkdir(exist_ok=True)
        
        # 下载基本信息文件
        info_files = ["UTTERANCEINFO.txt", "SPKINFO.txt"]
        for info_file in info_files:
            try:
                file_path = hf_hub_download(
                    repo_id="BAAI/SeniorTalk",
                    filename=info_file,
                    repo_type="dataset",
                    local_dir=str(asr_data_path)
                )
                logger.info(f"下载信息文件: {info_file}")
            except Exception as e:
                logger.warning(f"下载{info_file}失败: {e}")
        
        # 尝试下载sentence_data目录的内容
        try:
            sentence_path = hf_hub_download(
                repo_id="BAAI/SeniorTalk",
                filename="sentence_data",
                repo_type="dataset",
                local_dir=str(output_path),
                allow_patterns=["*.tar", "*.txt"]
            )
            logger.info(f"ASR数据下载到: {sentence_path}")
        except Exception as e:
            logger.warning(f"直接下载sentence_data失败: {e}")
            
            # 尝试下载完整数据集然后提取
            logger.info("尝试下载完整数据集...")
            repo_path = snapshot_download(
                repo_id="BAAI/SeniorTalk",
                repo_type="dataset",
                cache_dir=str(output_path / "cache"),
                local_dir=str(output_path / "seniortalk_full")
            )
            
            # 查找sentence_data目录
            full_path = Path(output_path / "seniortalk_full")
            sentence_data_source = None
            
            for item in full_path.rglob("sentence_data"):
                if item.is_dir():
                    sentence_data_source = item
                    break
            
            if sentence_data_source:
                # 复制sentence_data目录
                if asr_data_path.exists():
                    shutil.rmtree(asr_data_path)
                shutil.copytree(sentence_data_source, asr_data_path)
                logger.info(f"找到并复制ASR数据到: {asr_data_path}")
            else:
                logger.warning("未找到sentence_data目录")
        
        # 提取少量样本
        extract_asr_samples(str(asr_data_path), num_samples)
        
        # 创建数据清单
        manifest_data = {
            "dataset_name": "SeniorTalk ASR",
            "data_type": "ASR utterances",
            "num_samples": num_samples,
            "data_path": str(asr_data_path),
            "description": f"SeniorTalk ASR数据集的{num_samples}条测试样本"
        }
        
        manifest_path = output_path / "asr_dataset_info.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"ASR数据集信息保存到: {manifest_path}")
        
        return True
        
    except ImportError:
        logger.error("需要安装 huggingface_hub: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"下载失败: {e}")
        logger.info("请确保已正确登录HuggingFace并有权访问该数据集")
        return False

def extract_asr_samples(asr_data_path: str, num_samples: int = 5):
    """从ASR数据中提取少量样本"""
    
    asr_path = Path(asr_data_path)
    if not asr_path.exists():
        logger.error(f"ASR数据目录不存在: {asr_data_path}")
        return False
    
    # 创建样本目录
    sample_dir = asr_path / "samples"
    sample_dir.mkdir(exist_ok=True)
    
    # 查找wav文件夹中的tar文件
    wav_dir = asr_path / "wav"
    if wav_dir.exists():
        tar_files = list(wav_dir.rglob("*.tar"))
        
        if tar_files:
            # 从test集中提取样本（如果存在）
            test_tars = [f for f in tar_files if "test" in str(f)]
            if not test_tars:
                test_tars = tar_files[:1]  # 如果没有test文件，使用第一个
            
            for tar_file in test_tars[:1]:  # 只处理一个tar文件
                logger.info(f"正在从 {tar_file.name} 提取{num_samples}条样本...")
                
                try:
                    with tarfile.open(tar_file, 'r') as tar:
                        members = tar.getmembers()
                        wav_members = [m for m in members if m.isfile() and m.name.endswith('.wav')]
                        
                        # 提取指定数量的样本
                        for i, member in enumerate(wav_members[:num_samples]):
                            tar.extract(member, sample_dir)
                            logger.info(f"提取样本 {i+1}/{num_samples}: {member.name}")
                    
                    logger.info(f"✅ {num_samples}条ASR样本提取到: {sample_dir}")
                    return True
                    
                except Exception as e:
                    logger.error(f"提取样本失败: {e}")
                    return False
        else:
            logger.warning("未找到tar文件")
    
    # 查找现有的wav文件
    existing_wavs = list(asr_path.rglob("*.wav"))
    if existing_wavs:
        logger.info(f"找到{len(existing_wavs)}个现有的wav文件")
        for i, wav_file in enumerate(existing_wavs[:num_samples]):
            dest_file = sample_dir / f"sample_{i+1}_{wav_file.name}"
            shutil.copy2(wav_file, dest_file)
            logger.info(f"复制样本 {i+1}/{num_samples}: {wav_file.name}")
        return True
    
    logger.warning("未找到可提取的音频样本")
    return False

def main():
    parser = argparse.ArgumentParser(description="下载SeniorTalk ASR数据集测试样本")
    parser.add_argument("--output_dir", type=str, default="data/raw/audio",
                       help="输出目录 (默认: data/raw/audio)")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="要下载的样本数量 (默认: 5)")
    
    args = parser.parse_args()
    
    success = download_asr_samples(args.output_dir, args.num_samples)
    
    if success:
        logger.info(f"✅ SeniorTalk ASR数据集的{args.num_samples}条样本下载完成!")
        logger.info("现在可以查看 data/raw/audio/sentence_data/samples/ 目录中的样本文件")
    else:
        logger.error("❌ ASR数据集下载失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 