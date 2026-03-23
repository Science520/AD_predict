#!/usr/bin/env python3
"""
Integration example showing how to use audio augmentation with your AD prediction pipeline.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from audio_augmentation_for_ad import ADAudioAugmentor

def integrate_with_training_pipeline():
    """
    Example of how to integrate audio augmentation into your training pipeline.
    """
    print("Audio Augmentation Integration Example")
    print("=" * 50)
    
    # Initialize the augmentor
    augmentor = ADAudioAugmentor(sample_rate=16000)
    
    # Example: Augment training data
    print("1. Augmenting training data...")
    
    # Simulate your training data directory structure
    # Replace these paths with your actual data paths
    training_data_dir = "data/train"  # Your training audio files
    augmented_data_dir = "data/train_augmented"  # Where to save augmented files
    
    # Create augmented dataset (uncomment when you have real data)
    # augmentor.augment_dataset(training_data_dir, augmented_data_dir, num_augmentations=3)
    
    print("2. Creating augmentation pipeline for real-time use...")
    
    # For real-time augmentation during training
    def augment_batch(audio_batch, labels=None):
        """
        Augment a batch of audio samples.
        
        Args:
            audio_batch: List or array of audio samples
            labels: Optional labels (will be replicated for augmented samples)
            
        Returns:
            augmented_audio: List of augmented audio samples
            augmented_labels: List of corresponding labels (if provided)
        """
        augmented_audio = []
        augmented_labels = []
        
        for i, audio in enumerate(audio_batch):
            # Apply augmentation
            aug_audio = augmentor.augment_audio(audio)
            augmented_audio.append(aug_audio)
            
            # Replicate label for augmented sample
            if labels is not None:
                augmented_labels.append(labels[i])
        
        return augmented_audio, augmented_labels
    
    print("3. Example usage in training loop...")
    
    # Example training loop integration
    def training_step_with_augmentation(model, audio_batch, labels, optimizer):
        """
        Example training step with audio augmentation.
        """
        # Augment the batch
        augmented_audio, augmented_labels = augment_batch(audio_batch, labels)
        
        # Convert to tensors (adjust based on your framework)
        # augmented_audio_tensor = torch.tensor(augmented_audio)
        # augmented_labels_tensor = torch.tensor(augmented_labels)
        
        # Forward pass
        # predictions = model(augmented_audio_tensor)
        # loss = criterion(predictions, augmented_labels_tensor)
        
        # Backward pass
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        print(f"Processed batch of {len(augmented_audio)} augmented samples")
        return None  # Return actual loss in real implementation
    
    print("4. Creating different augmentation strategies...")
    
    # Light augmentation for validation/test
    light_augmentor = ADAudioAugmentor(sample_rate=16000)
    light_augmentor.augmentation_pipeline = light_augmentor.create_light_augmentation_pipeline()
    
    # Heavy augmentation for training
    heavy_augmentor = ADAudioAugmentor(sample_rate=16000)
    heavy_augmentor.augmentation_pipeline = heavy_augmentor.create_heavy_augmentation_pipeline()
    
    print("5. Example: Process single audio file...")
    
    # Example: Process a single audio file
    def process_audio_file(file_path, output_dir="augmented_samples"):
        """
        Process a single audio file with augmentation.
        """
        input_path = Path(file_path)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load audio
        audio, sr = librosa.load(input_path, sr=16000)
        
        # Apply different augmentation strategies
        light_augmented = light_augmentor.augment_audio(audio)
        heavy_augmented = heavy_augmentor.augment_audio(audio)
        
        # Save results
        stem = input_path.stem
        sf.write(output_path / f"{stem}_light.wav", light_augmented, 16000)
        sf.write(output_path / f"{stem}_heavy.wav", heavy_augmented, 16000)
        
        print(f"Processed {input_path.name}")
        return light_augmented, heavy_augmented
    
    # Test with a sample file (create one if needed)
    test_audio_path = "test_sample.wav"
    if not Path(test_audio_path).exists():
        # Create a test audio file
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        test_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        sf.write(test_audio_path, test_audio, sample_rate)
        print(f"Created test file: {test_audio_path}")
    
    # Process the test file
    light_aug, heavy_aug = process_audio_file(test_audio_path)
    
    print("\nIntegration example completed!")
    print("Key points:")
    print("- Use light augmentation for validation/test data")
    print("- Use heavy augmentation for training data")
    print("- Augment batches in real-time during training")
    print("- Save pre-augmented datasets for faster training")

def create_augmentation_config():
    """
    Create a configuration file for different augmentation strategies.
    """
    config = {
        "light_augmentation": {
            "description": "Minimal augmentation for validation/test",
            "transforms": [
                {"type": "AddGaussianNoise", "min_amplitude": 0.001, "max_amplitude": 0.005, "p": 0.3},
                {"type": "Gain", "min_gain_db": -2, "max_gain_db": 2, "p": 0.3}
            ]
        },
        "medium_augmentation": {
            "description": "Moderate augmentation for training",
            "transforms": [
                {"type": "AddGaussianNoise", "min_amplitude": 0.001, "max_amplitude": 0.01, "p": 0.4},
                {"type": "TimeStretch", "min_rate": 0.9, "max_rate": 1.1, "p": 0.3},
                {"type": "PitchShift", "min_semitones": -2, "max_semitones": 2, "p": 0.3},
                {"type": "Gain", "min_gain_db": -4, "max_gain_db": 4, "p": 0.4}
            ]
        },
        "heavy_augmentation": {
            "description": "Strong augmentation for training",
            "transforms": [
                {"type": "AddGaussianNoise", "min_amplitude": 0.005, "max_amplitude": 0.02, "p": 0.5},
                {"type": "RoomSimulator", "p": 0.4},
                {"type": "TimeStretch", "min_rate": 0.8, "max_rate": 1.2, "p": 0.4},
                {"type": "PitchShift", "min_semitones": -3, "max_semitones": 3, "p": 0.4},
                {"type": "Gain", "min_gain_db": -6, "max_gain_db": 6, "p": 0.5}
            ]
        }
    }
    
    import json
    with open("augmentation_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Created augmentation_config.json")
    return config

if __name__ == "__main__":
    integrate_with_training_pipeline()
    create_augmentation_config()



