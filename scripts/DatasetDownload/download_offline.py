#!/usr/bin/env python3
"""
Offline download script - tries multiple approaches
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

# Try different endpoints
endpoints = [
    'https://huggingface.co',
    'https://hf-mirror.com', 
    'https://huggingface.co',
]

def try_download_with_endpoint(endpoint, max_retries=3):
    """Try download with specific endpoint"""
    
    print(f"\n{'='*60}")
    print(f"Trying endpoint: {endpoint}")
    print(f"{'='*60}")
    
    os.environ['HF_ENDPOINT'] = endpoint
    configure_hf_env()
    
    for attempt in range(max_retries):
        print(f"\nAttempt {attempt + 1}/{max_retries}")
        
        try:
            from datasets import load_dataset
            
            print("Loading dataset...")
            ds = load_dataset(
                "evan0617/seniortalk", 
                "sentence_data",
                cache_dir=str(HF_CACHE_DIR),
                trust_remote_code=True
            )
            
            print("✅ Dataset loaded successfully!")
            return ds
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(10)
    
    return None

def main():
    """Main download function"""
    
    print("Starting offline download with multiple endpoints...")
    
    # Try each endpoint
    for endpoint in endpoints:
        ds = try_download_with_endpoint(endpoint)
        if ds is not None:
            print(f"\n🎉 Success with endpoint: {endpoint}")
            
            # Save dataset
            target_dir = Path("/data/AD_predict/data/raw/seniortalk_full")
            target_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Saving dataset to {target_dir}")
            ds.save_to_disk(str(target_dir))
            print("✅ Dataset saved successfully!")
            
            return True
    
    print("\n💥 All endpoints failed!")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
