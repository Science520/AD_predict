#!/usr/bin/env python3
"""
Download SeniorTalk sentence_data using datasets library
Saves audio files and transcripts to /data/AD_predict/data/raw/seniortalk_full/
"""

import os
import sys
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import soundfile as sf

# Target directory on the large disk
TARGET_DIR = Path("/data/AD_predict/data/raw/seniortalk_full")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# Create directory structure
SENTENCE_DATA_DIR = TARGET_DIR / "sentence_data"
SENTENCE_DATA_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("Downloading SeniorTalk sentence_data dataset")
print("=" * 80)
print(f"Target directory: {TARGET_DIR}")
print(f"\nDisk usage before download:")
os.system(f"df -h /data | tail -1")
print()

def is_valid_audio(audio_path, min_size=1000):
    """Check if audio file is valid (not empty or too small)"""
    if not audio_path.exists():
        return False
    if audio_path.stat().st_size < min_size:
        return False
    return True

def is_valid_transcript(txt_path, min_size=1):
    """Check if transcript file is valid"""
    if not txt_path.exists():
        return False
    if txt_path.stat().st_size < min_size:
        return False
    # Check if content is meaningful
    content = txt_path.read_text(encoding='utf-8').strip()
    if len(content) < 1:
        return False
    return True

def process_split(dataset, split_name):
    """Process one split of the dataset"""
    print(f"\n{'='*80}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'='*80}")
    
    # Create split directories
    audio_dir = SENTENCE_DATA_DIR / "audio" / split_name
    transcript_dir = SENTENCE_DATA_DIR / "transcript" / split_name
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total": len(dataset),
        "audio_saved": 0,
        "transcript_saved": 0,
        "invalid_removed": 0,
        "errors": 0
    }
    
    print(f"Total samples: {stats['total']}")
    
    for idx, sample in enumerate(tqdm(dataset, desc=f"Processing {split_name}")):
        try:
            # Get audio data
            audio = sample.get('audio')
            if audio is None:
                stats["errors"] += 1
                continue
            
            # Extract audio information
            audio_array = audio['array']
            sampling_rate = audio['sampling_rate']
            
            # Get the original filename or create one
            if 'path' in audio and audio['path']:
                original_filename = Path(audio['path']).name
                if not original_filename.endswith('.wav'):
                    original_filename = Path(audio['path']).stem + '.wav'
            else:
                original_filename = f"{split_name}_{idx:06d}.wav"
            
            # Save audio file
            audio_path = audio_dir / original_filename
            sf.write(str(audio_path), audio_array, sampling_rate)
            
            # Check if audio file is valid
            if not is_valid_audio(audio_path):
                audio_path.unlink()
                stats["invalid_removed"] += 1
                continue
            
            stats["audio_saved"] += 1
            
            # Get and save transcript
            transcript = sample.get('text', '').strip()
            if transcript:
                # Use same base name as audio
                txt_filename = audio_path.stem + '.txt'
                txt_path = transcript_dir / txt_filename
                txt_path.write_text(transcript, encoding='utf-8')
                
                # Check if transcript is valid
                if not is_valid_transcript(txt_path):
                    txt_path.unlink()
                    stats["invalid_removed"] += 1
                else:
                    stats["transcript_saved"] += 1
            
            # Print progress every 100 samples
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{stats['total']} samples...")
                
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            stats["errors"] += 1
            continue
    
    return stats

def download_and_save_metadata():
    """Try to download metadata files"""
    print(f"\n{'='*80}")
    print("Attempting to download metadata files")
    print(f"{'='*80}")
    
    try:
        from huggingface_hub import hf_hub_download
        
        metadata_files = ["SPKINFO.txt", "UTTERANCEINFO.txt"]
        
        for filename in metadata_files:
            try:
                print(f"\nDownloading {filename}...")
                downloaded_path = hf_hub_download(
                    repo_id="evan0617/seniortalk",
                    filename=filename,
                    repo_type="dataset",
                    local_dir=str(SENTENCE_DATA_DIR),
                    local_dir_use_symlinks=False
                )
                print(f"  ✓ {filename} downloaded to {downloaded_path}")
            except Exception as e:
                print(f"  ✗ Could not download {filename}: {e}")
                print(f"    You may need to download it manually from:")
                print(f"    https://huggingface.co/datasets/evan0617/seniortalk/tree/main")
        
    except Exception as e:
        print(f"Error downloading metadata: {e}")

def main():
    print("\nStep 1: Loading dataset from Hugging Face...")
    print("Note: First download may take a while and cache the dataset locally")
    print("Subsequent runs will be much faster.\n")
    
    try:
        # Load the dataset
        print("Loading sentence_data splits...")
        ds = load_dataset("evan0617/seniortalk", "sentence_data")
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"Available splits: {list(ds.keys())}")
        
        for split_name in ds.keys():
            print(f"  {split_name}: {len(ds[split_name])} samples")
        
        # Process each split
        all_stats = {}
        for split_name in ["train", "dev", "test"]:
            if split_name in ds:
                stats = process_split(ds[split_name], split_name)
                all_stats[split_name] = stats
            else:
                print(f"\nWarning: {split_name} split not found in dataset")
        
        # Try to download metadata
        download_and_save_metadata()
        
        # Print summary
        print(f"\n{'='*80}")
        print("DOWNLOAD SUMMARY")
        print(f"{'='*80}")
        
        total_audio = 0
        total_transcript = 0
        total_invalid = 0
        total_errors = 0
        
        for split_name, stats in all_stats.items():
            print(f"\n{split_name.upper()}:")
            print(f"  Total samples: {stats['total']}")
            print(f"  Audio files saved: {stats['audio_saved']}")
            print(f"  Transcript files saved: {stats['transcript_saved']}")
            print(f"  Invalid files removed: {stats['invalid_removed']}")
            print(f"  Errors: {stats['errors']}")
            
            total_audio += stats['audio_saved']
            total_transcript += stats['transcript_saved']
            total_invalid += stats['invalid_removed']
            total_errors += stats['errors']
        
        print(f"\nTOTAL:")
        print(f"  Audio files: {total_audio}")
        print(f"  Transcript files: {total_transcript}")
        print(f"  Invalid files removed: {total_invalid}")
        print(f"  Errors: {total_errors}")
        
        print(f"\nDisk usage after download:")
        os.system(f"df -h /data | tail -1")
        
        print(f"\nDirectory sizes:")
        os.system(f"du -sh {SENTENCE_DATA_DIR}/*")
        
        # Save stats to JSON
        stats_file = TARGET_DIR / "download_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                "splits": all_stats,
                "total": {
                    "audio_files": total_audio,
                    "transcript_files": total_transcript,
                    "invalid_removed": total_invalid,
                    "errors": total_errors
                }
            }, f, indent=2, ensure_ascii=False)
        print(f"\nStats saved to: {stats_file}")
        
        print("\n" + "="*80)
        print("✓ Download completed successfully!")
        print("="*80)
        print(f"\nData location: {SENTENCE_DATA_DIR}")
        
        print("\nDirectory structure:")
        os.system(f"tree -L 2 {SENTENCE_DATA_DIR} 2>/dev/null || find {SENTENCE_DATA_DIR} -maxdepth 2 -type d")
        
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("1. Verify SPKINFO.txt and UTTERANCEINFO.txt are present")
        print("   If not, download manually from:")
        print("   https://huggingface.co/datasets/evan0617/seniortalk/tree/main")
        print(f"   Place them in: {SENTENCE_DATA_DIR}")
        print("\n2. Inspect SPKINFO.txt to understand the speaker-to-province mapping")
        print("\n3. Run the data integration script to:")
        print("   - Map provinces to dialect labels")
        print("   - Merge with your existing labeled data")
        print("   - Prepare for training")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: {e}")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Check dependencies
    missing_deps = []
    
    try:
        import datasets
    except ImportError:
        missing_deps.append("datasets")
    
    try:
        import soundfile
    except ImportError:
        missing_deps.append("soundfile")
    
    try:
        import tqdm
    except ImportError:
        missing_deps.append("tqdm")
    
    if missing_deps:
        print("ERROR: Missing required packages:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install them with:")
        print(f"  pip install {' '.join(missing_deps)}")
        sys.exit(1)
    
    main()



