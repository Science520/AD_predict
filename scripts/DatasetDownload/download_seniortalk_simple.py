#!/usr/bin/env python3
"""
Simple script to download SeniorTalk sentence_data dataset
Downloads to /data/AD_predict/data/raw/seniortalk_full/
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# Target directory on the large disk
TARGET_DIR = Path("/data/AD_predict/data/raw/seniortalk_full")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Downloading SeniorTalk dataset from Hugging Face")
print("=" * 80)
print(f"Target directory: {TARGET_DIR}")
print(f"\nDisk usage before download:")
os.system(f"df -h /data | tail -1")
print()

def main():
    try:
        print("\nDownloading dataset files...")
        print("This will download all files from the repository.")
        print("This may take a while (dataset is ~30 hours of audio)...\n")
        
        # Download the entire dataset repository
        snapshot_download(
            repo_id="evan0617/seniortalk",
            repo_type="dataset",
            local_dir=str(TARGET_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4
        )
        
        print("\n" + "="*80)
        print("Download completed!")
        print("="*80)
        
        print(f"\nDisk usage after download:")
        os.system(f"df -h /data | tail -1")
        
        print(f"\nDirectory contents:")
        os.system(f"ls -lh {TARGET_DIR}")
        
        print(f"\nDirectory size:")
        os.system(f"du -sh {TARGET_DIR}")
        
        # Check for key files
        print(f"\nChecking for key files:")
        key_files = ["SPKINFO.txt", "UTTERANCEINFO.txt"]
        for filename in key_files:
            filepath = TARGET_DIR / filename
            if filepath.exists():
                print(f"  ✓ {filename} found")
            else:
                print(f"  ✗ {filename} NOT found")
        
        # Check for sentence_data directory
        sentence_data_dir = TARGET_DIR / "sentence_data"
        if sentence_data_dir.exists():
            print(f"\n  ✓ sentence_data directory found")
            print(f"    Contents:")
            os.system(f"ls -lh {sentence_data_dir}")
        else:
            print(f"\n  ✗ sentence_data directory NOT found")
        
        print("\n" + "="*80)
        print("Next steps:")
        print("="*80)
        print("1. Verify the downloaded files")
        print("2. Check SPKINFO.txt for speaker-to-province mapping")
        print("3. Run the integration script to merge with existing data")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: {e}")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Check if huggingface_hub is installed
    try:
        import huggingface_hub
    except ImportError:
        print("ERROR: huggingface_hub is not installed")
        print("Please install it with: pip install huggingface_hub")
        sys.exit(1)
    
    main()



