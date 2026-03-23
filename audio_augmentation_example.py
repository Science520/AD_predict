#!/usr/bin/env python3
"""
Audio Augmentation Example using audiomentations
This replaces the functionality of AudioAugmentor with direct audiomentations usage.
"""

import numpy as np
import librosa
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain, OneOf, SomeOf

def create_audio_augmentation_pipeline():
    """
    Create a comprehensive audio augmentation pipeline
    """
    augment = Compose([
        # Add noise with different probabilities
        OneOf([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.05, p=0.3),
        ]),
        
        # Time stretching (speed up/slow down)
        TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
        
        # Pitch shifting
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        
        # Gain adjustment
        Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
        
        # Apply some of these transforms randomly
        SomeOf([
            # Add more specific augmentations here if needed
        ], num_transforms=2, p=0.3)
    ])
    
    return augment

def augment_audio_file(input_path, output_path, sample_rate=16000):
    """
    Augment a single audio file
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save augmented audio
        sample_rate: Target sample rate
    """
    # Load audio
    audio, sr = librosa.load(input_path, sr=sample_rate)
    
    # Create augmentation pipeline
    augment = create_audio_augmentation_pipeline()
    
    # Apply augmentations
    augmented_audio = augment(samples=audio, sample_rate=sample_rate)
    
    # Save augmented audio
    sf.write(output_path, augmented_audio, sample_rate)
    
    print(f"Augmented audio saved to: {output_path}")
    return augmented_audio

def augment_audio_batch(input_dir, output_dir, sample_rate=16000):
    """
    Augment multiple audio files in a directory
    
    Args:
        input_dir: Directory containing input audio files
        output_dir: Directory to save augmented audio files
        sample_rate: Target sample rate
    """
    import os
    from pathlib import Path
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Get all audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_path.glob(f'*{ext}'))
        audio_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Create augmentation pipeline
    augment = create_audio_augmentation_pipeline()
    
    for audio_file in audio_files:
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=sample_rate)
            
            # Apply augmentations
            augmented_audio = augment(samples=audio, sample_rate=sample_rate)
            
            # Save augmented audio
            output_file = output_path / f"augmented_{audio_file.name}"
            sf.write(output_file, augmented_audio, sample_rate)
            
            print(f"Processed: {audio_file.name} -> {output_file.name}")
            
        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")

def create_simple_augmentation_pipeline():
    """
    Create a simple augmentation pipeline for quick testing
    """
    return Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5),
        Gain(min_gain_db=-3, max_gain_db=3, p=0.5),
    ])

if __name__ == "__main__":
    # Example usage
    print("Audio Augmentation Example")
    print("=" * 50)
    
    # Test with a simple pipeline
    augment = create_simple_augmentation_pipeline()
    
    # Create a test audio signal
    sample_rate = 16000
    duration = 2.0  # seconds
    frequency = 440  # Hz (A note)
    
    # Generate a simple sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    test_audio = np.sin(2 * np.pi * frequency * t)
    
    print(f"Original audio shape: {test_audio.shape}")
    print(f"Sample rate: {sample_rate}")
    
    # Apply augmentation
    augmented_audio = augment(samples=test_audio, sample_rate=sample_rate)
    
    print(f"Augmented audio shape: {augmented_audio.shape}")
    print("Audio augmentation pipeline created successfully!")
    
    # Save test files
    sf.write("test_original.wav", test_audio, sample_rate)
    sf.write("test_augmented.wav", augmented_audio, sample_rate)
    
    print("Test files saved: test_original.wav, test_augmented.wav")
