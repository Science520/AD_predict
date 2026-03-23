#!/usr/bin/env python3
"""
Audio Augmentation for AD (Alzheimer's Disease) Prediction
This script provides audio augmentation functionality specifically tailored for AD prediction tasks.
"""

import numpy as np
import librosa
import soundfile as sf
import os
from pathlib import Path
from audiomentations import (
    Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain, 
    OneOf, SomeOf, AddBackgroundNoise, RoomSimulator, 
    BandPassFilter, HighPassFilter, LowPassFilter
)

class ADAudioAugmentor:
    """
    Audio augmentor specifically designed for AD prediction tasks.
    Focuses on augmentations that preserve speech characteristics while adding variability.
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.augmentation_pipeline = self._create_ad_specific_pipeline()
    
    def _create_ad_specific_pipeline(self):
        """
        Create augmentation pipeline optimized for AD speech analysis.
        These augmentations preserve speech characteristics while adding realistic variability.
        """
        return Compose([
            # Noise augmentation - realistic for speech recordings
            OneOf([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.4),
                AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.02, p=0.2),
            ]),
            
            # Room simulation - adds realistic acoustic environment
            RoomSimulator(
                min_size_x=2.0, max_size_x=8.0,
                min_size_y=2.0, max_size_y=8.0,
                min_size_z=2.0, max_size_z=4.0,
                min_absorption_value=0.1, max_absorption_value=0.9,
                p=0.3
            ),
            
            # Frequency filtering - simulates different recording conditions
            OneOf([
                BandPassFilter(min_center_freq=300, max_center_freq=800, min_bandwidth_fraction=0.1, max_bandwidth_fraction=0.3, p=0.3),
                HighPassFilter(min_cutoff_freq=80, max_cutoff_freq=200, p=0.2),
                LowPassFilter(min_cutoff_freq=3000, max_cutoff_freq=8000, p=0.2),
            ]),
            
            # Subtle time stretching - preserves speech intelligibility
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
            
            # Subtle pitch shifting - maintains natural speech characteristics
            PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
            
            # Gain variation - simulates different recording levels
            Gain(min_gain_db=-4, max_gain_db=4, p=0.4),
        ])
    
    def augment_audio(self, audio):
        """
        Apply augmentations to audio data.
        
        Args:
            audio: numpy array of audio samples
            
        Returns:
            augmented_audio: numpy array of augmented audio samples
        """
        # Ensure audio is float32 for audiomentations
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Apply augmentations
        augmented_audio = self.augmentation_pipeline(samples=audio, sample_rate=self.sample_rate)
        
        return augmented_audio
    
    def augment_file(self, input_path, output_path):
        """
        Augment a single audio file.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save augmented audio file
        """
        # Load audio
        audio, sr = librosa.load(input_path, sr=self.sample_rate)
        
        # Apply augmentations
        augmented_audio = self.augment_audio(audio)
        
        # Save augmented audio
        sf.write(output_path, augmented_audio, self.sample_rate)
        
        return augmented_audio
    
    def augment_dataset(self, input_dir, output_dir, num_augmentations=3):
        """
        Augment all audio files in a directory with multiple variations.
        
        Args:
            input_dir: Directory containing input audio files
            output_dir: Directory to save augmented audio files
            num_augmentations: Number of augmented versions to create per file
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(f'*{ext}'))
            audio_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(audio_files)} audio files")
        print(f"Creating {num_augmentations} augmented versions per file")
        
        for audio_file in audio_files:
            try:
                # Load original audio
                audio, sr = librosa.load(audio_file, sr=self.sample_rate)
                
                # Create multiple augmented versions
                for i in range(num_augmentations):
                    # Apply augmentations
                    augmented_audio = self.augment_audio(audio)
                    
                    # Create output filename
                    stem = audio_file.stem
                    suffix = audio_file.suffix
                    output_filename = f"{stem}_aug_{i+1}{suffix}"
                    output_file = output_path / output_filename
                    
                    # Save augmented audio
                    sf.write(output_file, augmented_audio, self.sample_rate)
                    
                    print(f"Created: {output_filename}")
                
                # Also copy original file
                original_output = output_path / f"{audio_file.stem}_original{audio_file.suffix}"
                sf.write(original_output, audio, self.sample_rate)
                print(f"Copied original: {audio_file.stem}_original{audio_file.suffix}")
                
            except Exception as e:
                print(f"Error processing {audio_file.name}: {e}")
    
    def create_light_augmentation_pipeline(self):
        """
        Create a lighter augmentation pipeline for when you want minimal changes.
        """
        return Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.3),
            Gain(min_gain_db=-2, max_gain_db=2, p=0.3),
        ])
    
    def create_heavy_augmentation_pipeline(self):
        """
        Create a heavier augmentation pipeline for more aggressive data augmentation.
        """
        return Compose([
            OneOf([
                AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.02, p=0.5),
                AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.05, p=0.3),
            ]),
            RoomSimulator(
                min_size_x=3.0, max_size_x=10.0,
                min_size_y=3.0, max_size_y=10.0,
                min_size_z=2.5, max_size_z=5.0,
                min_absorption_value=0.2, max_absorption_value=0.8,
                p=0.4
            ),
            OneOf([
                BandPassFilter(min_center_freq=200, max_center_freq=1000, min_bandwidth_fraction=0.2, max_bandwidth_fraction=0.4, p=0.3),
                HighPassFilter(min_cutoff_freq=50, max_cutoff_freq=300, p=0.3),
                LowPassFilter(min_cutoff_freq=2000, max_cutoff_freq=10000, p=0.3),
            ]),
            TimeStretch(min_rate=0.8, max_rate=1.2, p=0.4),
            PitchShift(min_semitones=-3, max_semitones=3, p=0.4),
            Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
        ])

def main():
    """
    Example usage of the AD Audio Augmentor
    """
    print("AD Audio Augmentation Tool")
    print("=" * 50)
    
    # Initialize augmentor
    augmentor = ADAudioAugmentor(sample_rate=16000)
    
    # Create test audio (simulated speech-like signal)
    sample_rate = 16000
    duration = 3.0  # seconds
    
    # Generate a more complex test signal (simulating speech)
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a speech-like signal with multiple frequency components
    test_audio = (
        0.5 * np.sin(2 * np.pi * 200 * t) +  # Fundamental frequency
        0.3 * np.sin(2 * np.pi * 400 * t) +  # First harmonic
        0.2 * np.sin(2 * np.pi * 600 * t) +  # Second harmonic
        0.1 * np.sin(2 * np.pi * 800 * t)    # Third harmonic
    )
    
    # Add some envelope modulation to make it more speech-like
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)  # Slow modulation
    test_audio *= envelope
    
    print(f"Test audio shape: {test_audio.shape}")
    print(f"Sample rate: {sample_rate}")
    
    # Test different augmentation pipelines
    print("\nTesting augmentation pipelines...")
    
    # Light augmentation
    light_augmentor = ADAudioAugmentor(sample_rate)
    light_augmentor.augmentation_pipeline = light_augmentor.create_light_augmentation_pipeline()
    light_augmented = light_augmentor.augment_audio(test_audio)
    
    # Heavy augmentation
    heavy_augmentor = ADAudioAugmentor(sample_rate)
    heavy_augmentor.augmentation_pipeline = heavy_augmentor.create_heavy_augmentation_pipeline()
    heavy_augmented = heavy_augmentor.augment_audio(test_audio)
    
    # Save test files
    sf.write("ad_test_original.wav", test_audio, sample_rate)
    sf.write("ad_test_light_aug.wav", light_augmented, sample_rate)
    sf.write("ad_test_heavy_aug.wav", heavy_augmented, sample_rate)
    
    print("Test files saved:")
    print("- ad_test_original.wav (original)")
    print("- ad_test_light_aug.wav (light augmentation)")
    print("- ad_test_heavy_aug.wav (heavy augmentation)")
    
    print("\nAugmentation pipeline created successfully!")
    print("You can now use ADAudioAugmentor in your AD prediction pipeline.")

if __name__ == "__main__":
    main()



