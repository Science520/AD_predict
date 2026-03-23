#!/usr/bin/env python3
"""
Create a mock dataset for testing when network download fails
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict

def create_mock_seniortalk():
    """Create a mock SeniorTalk dataset for testing"""
    
    print("="*80)
    print("Creating mock SeniorTalk dataset")
    print("="*80)
    
    # Create target directory
    target_dir = Path("/data/AD_predict/data/raw/seniortalk_full")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock data
    mock_data = {
        'train': [
            {
                'text': 'Hello, how are you today?',
                'audio': {'path': 'mock_audio_1.wav', 'array': [0.1, 0.2, 0.3]},
                'speaker_id': 'speaker_001',
                'age': 65,
                'gender': 'female'
            },
            {
                'text': 'I am doing well, thank you for asking.',
                'audio': {'path': 'mock_audio_2.wav', 'array': [0.2, 0.3, 0.4]},
                'speaker_id': 'speaker_002', 
                'age': 72,
                'gender': 'male'
            },
            {
                'text': 'What did you have for breakfast?',
                'audio': {'path': 'mock_audio_3.wav', 'array': [0.3, 0.4, 0.5]},
                'speaker_id': 'speaker_001',
                'age': 65,
                'gender': 'female'
            }
        ],
        'validation': [
            {
                'text': 'The weather is nice today.',
                'audio': {'path': 'mock_audio_4.wav', 'array': [0.4, 0.5, 0.6]},
                'speaker_id': 'speaker_003',
                'age': 68,
                'gender': 'male'
            }
        ],
        'test': [
            {
                'text': 'I enjoy reading books in the evening.',
                'audio': {'path': 'mock_audio_5.wav', 'array': [0.5, 0.6, 0.7]},
                'speaker_id': 'speaker_004',
                'age': 70,
                'gender': 'female'
            }
        ]
    }
    
    # Create datasets
    datasets = {}
    for split_name, data in mock_data.items():
        datasets[split_name] = Dataset.from_list(data)
        print(f"Created {split_name} split with {len(data)} samples")
    
    # Create DatasetDict
    dataset_dict = DatasetDict(datasets)
    
    # Save dataset
    dataset_path = target_dir / "sentence_data"
    dataset_dict.save_to_disk(str(dataset_path))
    
    print(f"✅ Mock dataset saved to {dataset_path}")
    
    # Create dataset info
    info = {
        "dataset_name": "seniortalk_mock",
        "description": "Mock SeniorTalk dataset for testing",
        "splits": {
            "train": len(mock_data['train']),
            "validation": len(mock_data['validation']),
            "test": len(mock_data['test'])
        }
    }
    
    with open(target_dir / "dataset_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print("✅ Mock dataset created successfully!")
    return True

if __name__ == "__main__":
    print("Creating mock dataset for testing...")
    
    success = create_mock_seniortalk()
    
    if success:
        print("\n🎉 Mock dataset created!")
        print("Note: This is a mock dataset for testing purposes.")
        print("For production use, you'll need to download the real dataset.")
        sys.exit(0)
    else:
        print("\n💥 Mock dataset creation failed!")
        sys.exit(1)
