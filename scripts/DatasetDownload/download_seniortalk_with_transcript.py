#!/usr/bin/env python3
"""
下载SeniorTalk数据集（包含转录文本）
"""
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

def download_seniortalk_transcripts(output_dir: str = "/data/AD_predict/data/raw/seniortalk_full"):
    """
    下载SeniorTalk数据集的转录文本
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("📥 下载SeniorTalk转录文本")
    print("="*80)
    
    repo_id = "BAAI/SeniorTalk"
    
    try:
        # 尝试下载transcript目录
        print("\n尝试下载transcript目录...")
        
        # 列出可用的文件
        from huggingface_hub import list_repo_files
        
        print(f"📋 列出{repo_id}的文件...")
        files = list_repo_files(repo_id, repo_type="dataset")
        
        # 筛选transcript相关文件
        transcript_files = [f for f in files if 'transcript' in f.lower() or f.endswith('.txt')]
        
        print(f"\n找到{len(transcript_files)}个可能的转录文件:")
        for f in transcript_files[:20]:
            print(f"  - {f}")
        
        if not transcript_files:
            print("\n⚠️ 未找到转录文件")
            print("💡 SeniorTalk可能需要通过HuggingFace datasets库加载")
            return False
        
        # 下载转录文件
        print(f"\n📥 下载转录文件...")
        for file_path in transcript_files:
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    repo_type="dataset",
                    local_dir=str(output_path)
                )
                print(f"  ✅ {file_path}")
            except Exception as e:
                print(f"  ⚠️ 下载失败 {file_path}: {e}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n💡 可能的原因:")
        print("  1. 网络连接问题")
        print("  2. 需要登录HuggingFace: huggingface-cli login")
        print("  3. 需要申请数据集访问权限")
        return False


def try_load_with_datasets():
    """
    尝试使用datasets库加载SeniorTalk
    """
    print("\n" + "="*80)
    print("🔄 尝试使用datasets库加载（推荐方式）")
    print("="*80)
    
    try:
        from datasets import load_dataset
        
        print("\n📥 加载SeniorTalk数据集...")
        print("⏳ 这可能需要一些时间...")
        
        # 尝试加载训练集
        dataset = load_dataset("BAAI/SeniorTalk", split="train", streaming=True)
        
        print("\n✅ 数据集加载成功！")
        print("\n📊 数据集特征:")
        print(dataset.features)
        
        print("\n📝 前3个样本:")
        for i, sample in enumerate(dataset):
            if i >= 3:
                break
            print(f"\n样本 {i+1}:")
            print(f"  transcription: {sample.get('transcription', 'N/A')}")
            print(f"  speaker: {sample.get('speaker', 'N/A')}")
            print(f"  location: {sample.get('location', 'N/A')}")
        
        return True
        
    except ImportError:
        print("❌ datasets库未安装")
        print("   安装: pip install datasets")
        return False
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False


def main():
    print("="*80)
    print("🎯 SeniorTalk数据集下载（包含转录文本）")
    print("="*80)
    
    # 方式1: 使用datasets库（推荐）
    success = try_load_with_datasets()
    
    if not success:
        # 方式2: 直接下载文件
        print("\n尝试直接下载文件...")
        success = download_seniortalk_transcripts()
    
    if success:
        print("\n" + "="*80)
        print("✅ 下载成功！")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ 下载失败")
        print("="*80)
        print("\n💡 建议:")
        print("  1. 检查网络连接")
        print("  2. 登录HuggingFace: huggingface-cli login")
        print("  3. 访问 https://huggingface.co/datasets/BAAI/SeniorTalk 申请访问权限")
        print("\n⚠️ 如果无法获取转录文本，我们将使用Whisper-large-v3生成高质量伪标签")


if __name__ == "__main__":
    main()

