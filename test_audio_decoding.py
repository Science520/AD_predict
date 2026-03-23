#!/usr/bin/env python3
"""
快速测试音频解码功能
测试所有三种解码后端是否正常工作
"""

import sys
import io
from pathlib import Path
import pandas as pd

# 检查可用的音频库
print("="*70)
print("Testing Audio Decoding Backends")
print("="*70)

# 测试 soundfile
try:
    import soundfile as sf
    print(f"✓ soundfile: {sf.__version__}")
    SF_AVAILABLE = True
except ImportError as e:
    print(f"✗ soundfile: {e}")
    SF_AVAILABLE = False

# 测试 torchaudio
try:
    import torchaudio
    print(f"✓ torchaudio: {torchaudio.__version__}")
    TORCHAUDIO_AVAILABLE = True
except ImportError as e:
    print(f"✗ torchaudio: {e}")
    TORCHAUDIO_AVAILABLE = False

# 测试 librosa
try:
    import librosa
    print(f"✓ librosa: {librosa.__version__}")
    LIBROSA_AVAILABLE = True
except ImportError as e:
    print(f"✗ librosa: {e}")
    LIBROSA_AVAILABLE = False

print()

def decode_audio_bytes(audio_bytes, backend='auto'):
    """
    使用指定后端解码音频
    """
    if backend == 'torchaudio' and TORCHAUDIO_AVAILABLE:
        try:
            audio_tensor, sr = torchaudio.load(io.BytesIO(audio_bytes))
            audio_array = audio_tensor.numpy()
            if audio_array.shape[0] > 1:
                audio_array = audio_array.mean(axis=0)
            else:
                audio_array = audio_array[0]
            return audio_array, sr, 'torchaudio'
        except Exception as e:
            return None, None, f"torchaudio failed: {e}"
    
    elif backend == 'soundfile' and SF_AVAILABLE:
        try:
            audio_array, sr = sf.read(io.BytesIO(audio_bytes))
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            return audio_array, sr, 'soundfile'
        except Exception as e:
            return None, None, f"soundfile failed: {e}"
    
    elif backend == 'librosa' and LIBROSA_AVAILABLE:
        try:
            audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
            return audio_array, sr, 'librosa'
        except Exception as e:
            return None, None, f"librosa failed: {e}"
    
    # Auto mode: 尝试所有可用的后端
    elif backend == 'auto':
        for method in ['torchaudio', 'soundfile', 'librosa']:
            audio_array, sr, status = decode_audio_bytes(audio_bytes, backend=method)
            if audio_array is not None:
                return audio_array, sr, status
        return None, None, "All backends failed"
    
    return None, None, f"Backend {backend} not available"

# 测试实际数据
print("="*70)
print("Testing with SeniorTalk Data")
print("="*70)

SENIORTALK_TEST_DIR = Path("/data/AD_predict/data/raw/seniortalk_full/sentence_data")
test_files = list(SENIORTALK_TEST_DIR.glob("test-*.parquet"))

if not test_files:
    print(f"✗ No test files found in {SENIORTALK_TEST_DIR}")
    sys.exit(1)

print(f"Found {len(test_files)} test file(s)")
print(f"Testing first file: {test_files[0].name}\n")

try:
    df = pd.read_parquet(test_files[0])
    print(f"✓ Loaded parquet: {len(df)} samples")
    
    # 测试前5个样本
    test_count = min(5, len(df))
    print(f"\nTesting {test_count} samples with each backend:\n")
    
    results = {
        'torchaudio': {'success': 0, 'failed': 0},
        'soundfile': {'success': 0, 'failed': 0},
        'librosa': {'success': 0, 'failed': 0},
        'auto': {'success': 0, 'failed': 0}
    }
    
    for idx in range(test_count):
        row = df.iloc[idx]
        
        # 提取音频字节
        if hasattr(row, 'audio'):
            if isinstance(row.audio, dict) and 'bytes' in row.audio:
                audio_bytes = row.audio['bytes']
            elif isinstance(row.audio, bytes):
                audio_bytes = row.audio
            else:
                audio_bytes = bytes(row.audio)
        else:
            print(f"  Sample {idx}: No audio column")
            continue
        
        print(f"Sample {idx} (audio size: {len(audio_bytes)} bytes):")
        
        # 测试每个后端
        for backend in ['torchaudio', 'soundfile', 'librosa', 'auto']:
            audio_array, sr, status = decode_audio_bytes(audio_bytes, backend=backend)
            
            if audio_array is not None:
                results[backend]['success'] += 1
                print(f"  ✓ {backend:12s}: SR={sr}Hz, Length={len(audio_array)} samples")
            else:
                results[backend]['failed'] += 1
                print(f"  ✗ {backend:12s}: {status}")
        
        print()
    
    # 打印总结
    print("="*70)
    print("Summary")
    print("="*70)
    for backend, stats in results.items():
        total = stats['success'] + stats['failed']
        if total > 0:
            success_rate = (stats['success'] / total) * 100
            print(f"{backend:12s}: {stats['success']}/{total} success ({success_rate:.0f}%)")
    
    print("\n" + "="*70)
    print("Recommendation")
    print("="*70)
    
    if results['auto']['success'] == test_count:
        print("✅ Auto mode (multi-backend) successfully decoded all samples!")
        print("✅ The improved script should work well.")
    elif results['auto']['success'] > 0:
        print(f"⚠️  Auto mode decoded {results['auto']['success']}/{test_count} samples")
        print("   Some samples may fail, but evaluation can continue.")
    else:
        print("❌ All decoding attempts failed")
        print("   Need to investigate audio format issues")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

