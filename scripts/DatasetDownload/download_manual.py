#!/usr/bin/env python3
"""
Manual download script using direct file download
"""

import os
import sys
import requests
import json
from pathlib import Path
import time

def get_hf_token():
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HUGGINGFACE_HUB_TOKEN or HF_TOKEN before running this script.")
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    return token

def download_file_with_retry(url, output_path, max_retries=5):
    """Download a file with retry mechanism"""
    
    headers = {
        'Authorization': f'Bearer {get_hf_token()}',
        'User-Agent': 'datasets/2.16.0'
    }
    
    for attempt in range(max_retries):
        try:
            print(f"Downloading {url} (attempt {attempt + 1}/{max_retries})")
            
            response = requests.get(url, headers=headers, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Downloaded {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))  # Exponential backoff
    
    return False

def download_seniortalk_manual():
    """Manually download SeniorTalk dataset"""
    
    print("="*80)
    print("Manual SeniorTalk dataset download")
    print("="*80)
    
    # Create target directory
    target_dir = Path("/data/AD_predict/data/raw/seniortalk_full")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Base URL for the dataset
    base_url = "https://huggingface.co/datasets/evan0617/seniortalk/resolve/main"
    
    # Files to download for sentence_data split
    files_to_download = [
        "sentence_data/train/arrow-0.arrow",
        "sentence_data/train/arrow-1.arrow", 
        "sentence_data/train/arrow-2.arrow",
        "sentence_data/train/arrow-3.arrow",
        "sentence_data/train/arrow-4.arrow",
        "sentence_data/train/arrow-5.arrow",
        "sentence_data/train/arrow-6.arrow",
        "sentence_data/train/arrow-7.arrow",
        "sentence_data/train/arrow-8.arrow",
        "sentence_data/train/arrow-9.arrow",
        "sentence_data/train/arrow-10.arrow",
        "sentence_data/train/arrow-11.arrow",
        "sentence_data/train/arrow-12.arrow",
        "sentence_data/train/arrow-13.arrow",
        "sentence_data/train/arrow-14.arrow",
        "sentence_data/train/arrow-15.arrow",
        "sentence_data/validation/arrow-0.arrow",
        "sentence_data/test/arrow-0.arrow",
        "sentence_data/dataset_info.json",
        "sentence_data/state.json"
    ]
    
    success_count = 0
    total_files = len(files_to_download)
    
    for file_path in files_to_download:
        url = f"{base_url}/{file_path}"
        output_path = target_dir / file_path
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if download_file_with_retry(url, output_path):
            success_count += 1
        else:
            print(f"❌ Failed to download {file_path}")
    
    print(f"\n📊 Download summary: {success_count}/{total_files} files downloaded")
    
    if success_count == total_files:
        print("✅ All files downloaded successfully!")
        return True
    else:
        print("⚠️ Some files failed to download")
        return False

if __name__ == "__main__":
    print("Starting manual download...")
    
    success = download_seniortalk_manual()
    
    if success:
        print("\n🎉 Manual download completed!")
        sys.exit(0)
    else:
        print("\n💥 Manual download failed!")
        sys.exit(1)
