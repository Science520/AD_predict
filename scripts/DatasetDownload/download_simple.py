#!/usr/bin/env python3
"""
Simple download script with network optimizations
"""

import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
HF_CACHE_DIR = ROOT_DIR / "hf_cache"

def configure_hf_env():
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HUGGINGFACE_HUB_TOKEN or HF_TOKEN before running this script.")
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))

# Network optimizations
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'  # 10 minutes timeout
os.environ['HF_HUB_DOWNLOAD_RETRIES'] = '20'   # 20 retries
os.environ['HF_HUB_ETAG_TIMEOUT'] = '30'       # 30 seconds for ETag

def download_dataset():
    """Download the dataset with optimizations"""
    
    print("="*80)
    print("Starting SeniorTalk dataset download")
    print("="*80)
    
    try:
        from datasets import load_dataset
        configure_hf_env()
        
        print("Loading dataset from Hugging Face...")
        print("This may take a while for the first download...")
        
        # Load dataset with optimizations
        ds = load_dataset(
            "evan0617/seniortalk", 
            "sentence_data",
            cache_dir=str(HF_CACHE_DIR),
            download_mode="reuse_dataset_if_exists"
        )
        
        print("✅ Dataset loaded successfully!")
        print(f"Dataset splits: {list(ds.keys())}")
        
        # Print dataset info
        for split_name, split_data in ds.items():
            print(f"  {split_name}: {len(split_data)} samples")
        
        # Save to target directory
        target_dir = Path("/data/AD_predict/data/raw/seniortalk_full")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving dataset to {target_dir}")
        ds.save_to_disk(str(target_dir))
        print("✅ Dataset saved successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting download with network optimizations...")
    
    # Try multiple times with different settings
    for attempt in range(3):
        print(f"\nAttempt {attempt + 1}/3")
        
        if attempt > 0:
            # Try different endpoint on retry
            if attempt == 1:
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                print("Trying with mirror endpoint...")
            else:
                os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
                print("Trying with original endpoint...")
        
        success = download_dataset()
        if success:
            print("\n🎉 Download completed successfully!")
            sys.exit(0)
        else:
            print(f"❌ Attempt {attempt + 1} failed")
            if attempt < 2:
                print("⏳ Waiting 30 seconds before retry...")
                time.sleep(30)
    
    print("\n💥 All attempts failed!")
    sys.exit(1)
