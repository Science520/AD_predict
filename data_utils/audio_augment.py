"""
音频数据增强模块
用于对样本数量较少的方言类别进行数据增强
"""

import librosa
import soundfile as sf
import numpy as np
from audiomentations import (
    Compose, 
    AddGaussianNoise, 
    TimeStretch, 
    PitchShift,
    Shift
)
import os
from pathlib import Path


class AudioAugmentor:
    """音频增强器"""
    
    def __init__(self, sample_rate=16000):
        """
        初始化音频增强器
        
        Args:
            sample_rate: 采样率，Whisper使用16000Hz
        """
        self.sample_rate = sample_rate
        
        # 定义多个增强管道，用于生成不同的增强版本
        self.augment_pipelines = [
            # 管道1: 轻微变速 + 噪声
            Compose([
                TimeStretch(min_rate=0.95, max_rate=1.05, p=1.0),
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.5),
            ]),
            
            # 管道2: 音调变化 + 时移
            Compose([
                PitchShift(min_semitones=-1, max_semitones=1, p=1.0),
                Shift(min_shift=-0.2, max_shift=0.2, p=0.5),
            ]),
            
            # 管道3: 变速 + 音调 + 噪声组合
            Compose([
                TimeStretch(min_rate=0.97, max_rate=1.03, p=1.0),
                PitchShift(min_semitones=-0.5, max_semitones=0.5, p=1.0),
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.003, p=0.3),
            ]),
            
            # 管道4: 时间偏移 + 轻微噪声
            Compose([
                Shift(min_shift=-0.3, max_shift=0.3, p=1.0),
                AddGaussianNoise(min_amplitude=0.002, max_amplitude=0.006, p=0.4),
            ]),
        ]
    
    def augment_audio_file(self, audio_path, num_augmentations=4):
        """
        对单个音频文件进行增强
        
        Args:
            audio_path: 音频文件路径
            num_augmentations: 生成的增强版本数量
            
        Returns:
            augmented_audios: 增强后的音频数据列表 [(audio_array, sample_rate), ...]
        """
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        augmented_audios = []
        
        # 使用不同的管道生成增强版本
        for i in range(num_augmentations):
            pipeline_idx = i % len(self.augment_pipelines)
            pipeline = self.augment_pipelines[pipeline_idx]
            
            # 应用增强
            augmented = pipeline(samples=audio, sample_rate=self.sample_rate)
            augmented_audios.append((augmented, self.sample_rate))
        
        return augmented_audios
    
    def augment_and_save(self, audio_path, output_dir, base_name, num_augmentations=4):
        """
        对音频进行增强并保存到文件
        
        Args:
            audio_path: 原始音频路径
            output_dir: 输出目录
            base_name: 基础文件名
            num_augmentations: 增强版本数量
            
        Returns:
            saved_paths: 保存的增强音频路径列表
        """
        os.makedirs(output_dir, exist_ok=True)
        
        augmented_audios = self.augment_audio_file(audio_path, num_augmentations)
        
        saved_paths = []
        for i, (audio, sr) in enumerate(augmented_audios):
            output_path = os.path.join(output_dir, f"{base_name}_aug_{i}.wav")
            sf.write(output_path, audio, sr)
            saved_paths.append(output_path)
        
        return saved_paths


def augment_audio_in_memory(audio_path, num_augmentations=4, sample_rate=16000):
    """
    在内存中对音频进行增强（不保存文件）
    
    Args:
        audio_path: 音频文件路径
        num_augmentations: 增强版本数量
        sample_rate: 采样率
        
    Returns:
        augmented_audios: 增强后的音频数据列表
    """
    augmentor = AudioAugmentor(sample_rate=sample_rate)
    return augmentor.augment_audio_file(audio_path, num_augmentations)


if __name__ == "__main__":
    # 测试代码
    print("音频增强模块测试")
    
    # 示例：对一个音频文件进行增强
    test_audio = "/data/AD_predict/data/raw/audio/elderly_audios/elderly_audio_0001.wav"
    
    if os.path.exists(test_audio):
        augmentor = AudioAugmentor()
        augmented = augmentor.augment_audio_file(test_audio, num_augmentations=2)
        print(f"原始音频: {test_audio}")
        print(f"生成了 {len(augmented)} 个增强版本")
        
        # 测试保存功能
        saved_paths = augmentor.augment_and_save(
            test_audio, 
            "./temp_augmented",
            "test_audio",
            num_augmentations=2
        )
        print(f"增强音频已保存到: {saved_paths}")
    else:
        print(f"测试音频文件不存在: {test_audio}")


