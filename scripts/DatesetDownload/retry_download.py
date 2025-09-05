#!/usr/bin/env python3
"""
智能重试下载HuggingFace数据集
"""
import time
import logging
from pathlib import Path
import requests
from huggingface_hub import snapshot_download, hf_hub_download
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def wait_and_retry_download(repo_id, max_retries=5, wait_time=300):
    """
    等待并重试下载数据集
    
    Args:
        repo_id: 数据集ID，如 "BAAI/SeniorTalk"
        max_retries: 最大重试次数
        wait_time: 等待时间（秒）
    """
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 尝试下载 {repo_id} (第 {attempt + 1}/{max_retries} 次)")
            
            # 先检查数据集状态
            status = check_dataset_status(repo_id)
            logger.info(f"📊 数据集状态: {status}")
            
            if status == "error":
                logger.warning("❌ 数据集当前有错误，等待修复...")
                if attempt < max_retries - 1:
                    logger.info(f"⏰ 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
            
            # 尝试下载
            path = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                cache_dir="data/raw/audio/cache",
                local_dir="data/raw/audio/seniortalk_download",
                resume_download=True  # 支持断点续传
            )
            
            logger.info(f"✅ 下载成功: {path}")
            return path
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ 下载失败: {error_msg}")
            
            if "Job manager crashed" in error_msg:
                logger.warning("🔧 检测到JobManager崩溃错误，这通常是服务器端临时问题")
                
            elif "missing heartbeats" in error_msg:
                logger.warning("💓 检测到心跳丢失错误，网络连接可能不稳定")
                
            if attempt < max_retries - 1:
                wait_time_current = wait_time * (2 ** attempt)  # 指数退避
                logger.info(f"⏰ 等待 {wait_time_current} 秒后重试...")
                time.sleep(wait_time_current)
            else:
                logger.error("❌ 所有重试都失败了")
                return None

def check_dataset_status(repo_id):
    """检查数据集状态"""
    try:
        # 尝试访问数据集API
        api_url = f"https://datasets-server.huggingface.co/info?dataset={repo_id}"
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 200:
            return "ok"
        elif response.status_code >= 500:
            return "server_error"
        else:
            return "error"
            
    except Exception as e:
        logger.warning(f"无法检查数据集状态: {e}")
        return "unknown"

def download_alternative_datasets():
    """下载替代数据集"""
    
    alternative_datasets = [
        {
            "name": "THCHS30",
            "repo_id": "speech-lab/THCHS-30",
            "description": "清华大学免费中文语音数据集"
        },
        {
            "name": "WenetSpeech",
            "repo_id": "wenet-e2e/wenetspeech", 
            "description": "10000+小时中文语音数据集"
        },
        {
            "name": "Adult Voice",
            "repo_id": "longmaodata/Adult-Voice",
            "description": "成人语音数据集"
        }
    ]
    
    logger.info("📦 尝试下载替代数据集...")
    
    for dataset in alternative_datasets:
        try:
            logger.info(f"🔄 下载 {dataset['name']}: {dataset['description']}")
            
            path = snapshot_download(
                repo_id=dataset["repo_id"],
                repo_type="dataset",
                cache_dir="data/raw/audio/cache",
                local_dir=f"data/raw/audio/{dataset['name'].lower()}",
                resume_download=True
            )
            
            logger.info(f"✅ {dataset['name']} 下载成功: {path}")
            return path
            
        except Exception as e:
            logger.warning(f"⚠️ {dataset['name']} 下载失败: {e}")
            continue
    
    return None

def main():
    print("🚀 HuggingFace数据集智能重试下载工具")
    print("="*60)
    
    # 首先尝试下载SeniorTalk
    seniortalk_path = wait_and_retry_download("BAAI/SeniorTalk")
    
    if seniortalk_path:
        print(f"\n🎉 SeniorTalk下载成功: {seniortalk_path}")
    else:
        print("\n⚠️ SeniorTalk下载失败，尝试替代数据集...")
        alt_path = download_alternative_datasets()
        
        if alt_path:
            print(f"✅ 替代数据集下载成功: {alt_path}")
        else:
            print("❌ 所有数据集下载都失败了")
            print("\n💡 建议:")
            print("1. 检查网络连接")
            print("2. 联系HuggingFace支持")
            print("3. 稍后再试")

if __name__ == "__main__":
    main() 