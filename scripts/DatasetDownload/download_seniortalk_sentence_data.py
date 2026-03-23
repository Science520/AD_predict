#!/usr/bin/env python3
"""
Download SeniorTalk sentence_data dataset from Hugging Face
Downloads to /data/AD_predict/data/raw/seniortalk_full/
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import tarfile

# Target directory on the large disk
TARGET_DIR = Path("/data/AD_predict/data/raw/seniortalk_full")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# Create subdirectories
SENTENCE_DATA_DIR = TARGET_DIR / "sentence_data"
SENTENCE_DATA_DIR.mkdir(exist_ok=True)

WAV_DIR = SENTENCE_DATA_DIR / "wav"
TRANSCRIPT_DIR = SENTENCE_DATA_DIR / "transcript"
WAV_DIR.mkdir(exist_ok=True)
TRANSCRIPT_DIR.mkdir(exist_ok=True)

# Create train/dev/test subdirectories
for split in ["train", "dev", "test"]:
    (WAV_DIR / split).mkdir(exist_ok=True)
    (TRANSCRIPT_DIR / split).mkdir(exist_ok=True)

print("=" * 80)
print("Downloading SeniorTalk sentence_data dataset")
print("=" * 80)
print(f"Target directory: {TARGET_DIR}")
print(f"Disk usage before download:")
os.system(f"df -h /data | tail -1")
print()

def extract_tar_file(tar_path, extract_to):
    """Extract tar file and return list of extracted files"""
    print(f"  Extracting {tar_path.name}...")
    try:
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(extract_to)
            members = tar.getnames()
        return members
    except Exception as e:
        print(f"  Error extracting {tar_path}: {e}")
        return []

def is_empty_or_invalid_file(file_path, min_size=100):
    """Check if file is empty or too small to be valid"""
    if not file_path.exists():
        return True
    if file_path.stat().st_size < min_size:
        return True
    return False

def process_split(dataset_split, split_name):
    """Process one split (train/dev/test) of the dataset"""
    print(f"\n{'='*80}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'='*80}")
    print(f"Total samples: {len(dataset_split)}")
    
    stats = {
        "total": len(dataset_split),
        "audio_saved": 0,
        "audio_extracted": 0,
        "transcript_saved": 0,
        "empty_files_removed": 0,
        "errors": 0
    }
    
    wav_split_dir = WAV_DIR / split_name
    transcript_split_dir = TRANSCRIPT_DIR / split_name
    
    # Process each sample
    for idx, sample in enumerate(tqdm(dataset_split, desc=f"Processing {split_name}")):
        try:
            # Get audio data
            audio_data = sample.get('audio')
            if audio_data is None:
                stats["errors"] += 1
                continue
            
            # Get file path and transcript
            audio_path = audio_data.get('path', f'unknown_{idx}.wav')
            audio_array = audio_data.get('array')
            sampling_rate = audio_data.get('sampling_rate', 16000)
            
            # Get transcript
            transcript = sample.get('text', '')
            
            # Save audio file
            # If it's a tar file path, we need to handle it differently
            if audio_path.endswith('.tar'):
                # This is a tar archive, we'll handle it specially
                tar_name = Path(audio_path).name
                tar_save_path = wav_split_dir / tar_name
                
                # Check if we have the actual audio array
                if audio_array is not None:
                    import soundfile as sf
                    # Extract filename from path
                    wav_filename = Path(audio_path).stem + '.wav'
                    wav_save_path = wav_split_dir / wav_filename
                    sf.write(wav_save_path, audio_array, sampling_rate)
                    stats["audio_saved"] += 1
                    
                    # Check if file is valid
                    if is_empty_or_invalid_file(wav_save_path):
                        wav_save_path.unlink()
                        stats["empty_files_removed"] += 1
                        continue
            else:
                # Regular wav file
                if audio_array is not None:
                    import soundfile as sf
                    wav_filename = Path(audio_path).name
                    if not wav_filename.endswith('.wav'):
                        wav_filename = Path(audio_path).stem + '.wav'
                    wav_save_path = wav_split_dir / wav_filename
                    sf.write(wav_save_path, audio_array, sampling_rate)
                    stats["audio_saved"] += 1
                    
                    # Check if file is valid
                    if is_empty_or_invalid_file(wav_save_path):
                        wav_save_path.unlink()
                        stats["empty_files_removed"] += 1
                        continue
            
            # Save transcript
            if transcript:
                # Use the same base name as audio file
                if 'wav_save_path' in locals():
                    txt_filename = wav_save_path.stem + '.txt'
                else:
                    txt_filename = f'sample_{idx}.txt'
                
                txt_save_path = transcript_split_dir / txt_filename
                txt_save_path.write_text(transcript, encoding='utf-8')
                stats["transcript_saved"] += 1
                
                # Check if transcript file is valid
                if is_empty_or_invalid_file(txt_save_path, min_size=1):
                    txt_save_path.unlink()
                    stats["empty_files_removed"] += 1
            
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            stats["errors"] += 1
            continue
    
    return stats

def download_metadata_files():
    """Download SPKINFO.txt and UTTERANCEINFO.txt"""
    print(f"\n{'='*80}")
    print("Downloading metadata files")
    print(f"{'='*80}")
    
    try:
        # Load dataset to get metadata
        print("Loading dataset metadata...")
        ds = load_dataset("evan0617/seniortalk", "sentence_data", split="train", streaming=True)
        
        # Try to get metadata from dataset info
        dataset_info = load_dataset("evan0617/seniortalk", "sentence_data")
        
        # Check if metadata files are available in the dataset
        # We'll need to manually download these from the repo
        print("\nNote: SPKINFO.txt and UTTERANCEINFO.txt need to be downloaded manually")
        print("Please download them from: https://huggingface.co/datasets/evan0617/seniortalk")
        print(f"And place them in: {SENTENCE_DATA_DIR}")
        
    except Exception as e:
        print(f"Error accessing metadata: {e}")

def main():
    print("\nStep 1: Loading dataset from Hugging Face...")
    print("This may take a while for the first download...")
    
    try:
        # Load the dataset
        print("\nLoading sentence_data...")
        ds = load_dataset("evan0617/seniortalk", "sentence_data")
        
        print(f"\nDataset loaded successfully!")
        print(f"Available splits: {list(ds.keys())}")
        for split_name in ds.keys():
            print(f"  {split_name}: {len(ds[split_name])} samples")
        
        # Process each split
        all_stats = {}
        for split_name in ["train", "dev", "test"]:
            if split_name in ds:
                stats = process_split(ds[split_name], split_name)
                all_stats[split_name] = stats
        
        # Download metadata
        download_metadata_files()
        
        # Print summary
        print(f"\n{'='*80}")
        print("DOWNLOAD SUMMARY")
        print(f"{'='*80}")
        for split_name, stats in all_stats.items():
            print(f"\n{split_name.upper()}:")
            print(f"  Total samples: {stats['total']}")
            print(f"  Audio files saved: {stats['audio_saved']}")
            print(f"  Transcript files saved: {stats['transcript_saved']}")
            print(f"  Empty files removed: {stats['empty_files_removed']}")
            print(f"  Errors: {stats['errors']}")
        
        print(f"\nDisk usage after download:")
        os.system(f"df -h /data | tail -1")
        
        print(f"\nDirectory structure:")
        os.system(f"du -sh {TARGET_DIR}/*")
        
        # Save stats to JSON
        stats_file = TARGET_DIR / "download_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nStats saved to: {stats_file}")
        
        print("\n" + "="*80)
        print("Download completed successfully!")
        print("="*80)
        print(f"\nData location: {TARGET_DIR}")
        print("\nNext steps:")
        print("1. Download SPKINFO.txt and UTTERANCEINFO.txt manually from:")
        print("   https://huggingface.co/datasets/evan0617/seniortalk/tree/main")
        print(f"2. Place them in: {SENTENCE_DATA_DIR}")
        print("3. Run the data integration script to merge with existing data")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: {e}")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Check if soundfile is installed
    try:
        import soundfile
    except ImportError:
        print("ERROR: soundfile is not installed")
        print("Please install it with: pip install soundfile")
        sys.exit(1)
    
    main()



