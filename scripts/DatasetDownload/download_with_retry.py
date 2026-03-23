#!/usr/bin/env python3
"""
Robust download script with retry mechanism and network optimization
"""

import os
import sys
import time
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
HF_CACHE_DIR = ROOT_DIR / "hf_cache"

def configure_hf_env():
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HUGGINGFACE_HUB_TOKEN or HF_TOKEN before running this script.")
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))

# Set environment variables for better network handling
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Add timeout and retry settings
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5 minutes timeout
os.environ['HF_HUB_DOWNLOAD_RETRIES'] = '10'   # 10 retries

def download_with_retry(max_retries=5, delay=30):
    """Download with retry mechanism"""
    
    for attempt in range(max_retries):
        print(f"\n{'='*80}")
        print(f"Download attempt {attempt + 1}/{max_retries}")
        print(f"{'='*80}")
        
        try:
            # Import here to ensure environment variables are set
            from datasets import load_dataset
            configure_hf_env()
            
            print("Loading dataset from Hugging Face...")
            print("Note: This may take a while for the first download...")
            
            # Load the dataset with timeout settings
            ds = load_dataset(
                "evan0617/seniortalk", 
                "sentence_data",
                cache_dir=str(HF_CACHE_DIR)
            )
            
            print("✅ Dataset loaded successfully!")
            print(f"Dataset info: {ds}")
            
            # Save to target directory
            target_dir = Path("/data/AD_predict/data/raw/seniortalk_full")
            target_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Saving dataset to {target_dir}")
            ds.save_to_disk(str(target_dir))
            print("✅ Dataset saved successfully!")
            
            return True
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                print(f"⏳ Waiting {delay} seconds before retry...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print("❌ All attempts failed!")
                return False
    
    return False

if __name__ == "__main__":
    print("Starting robust download with retry mechanism...")
    success = download_with_retry()
    
    if success:
        print("\n🎉 Download completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Download failed after all retries!")
        sys.exit(1)
