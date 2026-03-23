#!/usr/bin/env python3
"""
Test the created dataset
"""

from datasets import load_from_disk

def test_dataset():
    """Test the dataset loading"""
    
    print("Testing dataset loading...")
    
    try:
        ds = load_from_disk('/data/AD_predict/data/raw/seniortalk_full/sentence_data')
        print("✅ Dataset loaded successfully!")
        print(f"Splits: {list(ds.keys())}")
        print(f"Train samples: {len(ds['train'])}")
        print(f"Validation samples: {len(ds['validation'])}")
        print(f"Test samples: {len(ds['test'])}")
        
        # Show a sample
        print("\nSample from train set:")
        sample = ds['train'][0]
        print(f"Text: {sample['text']}")
        print(f"Speaker ID: {sample['speaker_id']}")
        print(f"Age: {sample['age']}")
        print(f"Gender: {sample['gender']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_dataset()
    if success:
        print("\n🎉 Dataset test successful!")
    else:
        print("\n💥 Dataset test failed!")

