#!/usr/bin/env python3
"""
声学特征提取器 - 用于PMM患者分层

从语音信号中提取关键的声学特征，如基频、语速、停顿等
"""

import numpy as np
import librosa
import torch
from typing import Dict, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AcousticFeatureExtractor:
    """
    声学特征提取器
    
    提取用于患者分层的声学特征：
    - 基频 (F0) 统计量
    - 语速相关特征
    - 停顿特征
    - 能量特征
    - 谱特征
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def extract_f0_features(self, audio: np.ndarray) -> Dict[str, float]:
        """提取基频特征"""
        
        # 使用pyin算法提取基频
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),  # 约65Hz
            fmax=librosa.note_to_hz('C7'),   # 约2093Hz
            sr=self.sample_rate
        )
        
        # 过滤掉无声部分
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) == 0:
            return {
                'f0_mean': 0.0,
                'f0_std': 0.0,
                'f0_range': 0.0,
                'f0_median': 0.0
            }
        
        return {
            'f0_mean': np.nanmean(f0_voiced),
            'f0_std': np.nanstd(f0_voiced),
            'f0_range': np.nanmax(f0_voiced) - np.nanmin(f0_voiced),
            'f0_median': np.nanmedian(f0_voiced)
        }
    
    def extract_energy_features(self, audio: np.ndarray) -> Dict[str, float]:
        """提取能量特征"""
        
        # RMS能量
        rms = librosa.feature.rms(y=audio)[0]
        
        # 短时能量
        frame_length = int(0.025 * self.sample_rate)
        hop_length = int(0.01 * self.sample_rate)
        
        energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        return {
            'energy_mean': np.mean(energy),
            'energy_std': np.std(energy),
            'energy_max': np.max(energy),
            'energy_dynamic_range': np.max(energy) - np.min(energy)
        }
    
    def extract_pause_features(self, audio: np.ndarray) -> Dict[str, float]:
        """提取停顿特征"""
        
        # 计算能量
        frame_length = int(0.025 * self.sample_rate)
        hop_length = int(0.01 * self.sample_rate)
        
        energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # 检测静音段（能量低于平均值的10%）
        silence_threshold = np.mean(energy) * 0.1
        silence_mask = energy < silence_threshold
        
        # 统计停顿
        time_frames = librosa.frames_to_time(
            np.arange(len(energy)),
            sr=self.sample_rate,
            hop_length=hop_length
        )
        
        # 找连续的静音段
        pauses = []
        in_pause = False
        pause_start = 0
        
        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_pause:
                pause_start = time_frames[i]
                in_pause = True
            elif not is_silent and in_pause:
                pause_duration = time_frames[i] - pause_start
                if pause_duration >= 0.3:  # 至少300ms才算停顿
                    pauses.append(pause_duration)
                in_pause = False
        
        audio_duration = len(audio) / self.sample_rate
        
        if len(pauses) == 0:
            return {
                'pause_count': 0,
                'pause_total_time': 0.0,
                'pause_ratio': 0.0,
                'pause_mean_duration': 0.0,
                'pause_frequency': 0.0
            }
        
        return {
            'pause_count': len(pauses),
            'pause_total_time': sum(pauses),
            'pause_ratio': sum(pauses) / audio_duration,
            'pause_mean_duration': np.mean(pauses),
            'pause_frequency': len(pauses) / audio_duration  # 每秒停顿次数
        }
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """提取谱特征"""
        
        # 谱质心
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, 
            sr=self.sample_rate
        )[0]
        
        # 谱带宽
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, 
            sr=self.sample_rate
        )[0]
        
        # 谱对比度
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio, 
            sr=self.sample_rate
        )
        
        # 谱滚降
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, 
            sr=self.sample_rate
        )[0]
        
        return {
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_contrast_mean': np.mean(spectral_contrast),
            'spectral_rolloff_mean': np.mean(spectral_rolloff)
        }
    
    def extract_mfcc_features(self, audio: np.ndarray, n_mfcc: int = 13) -> Dict[str, float]:
        """提取MFCC特征"""
        
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate, 
            n_mfcc=n_mfcc
        )
        
        features = {}
        for i in range(n_mfcc):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            
        return features
    
    def extract_all_features(
        self, 
        audio_path: str,
        include_mfcc: bool = False
    ) -> np.ndarray:
        """
        提取所有声学特征
        
        Args:
            audio_path: 音频文件路径
            include_mfcc: 是否包含MFCC特征（会增加特征维度）
            
        Returns:
            特征向量
        """
        try:
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # 提取各类特征
            f0_features = self.extract_f0_features(audio)
            energy_features = self.extract_energy_features(audio)
            pause_features = self.extract_pause_features(audio)
            spectral_features = self.extract_spectral_features(audio)
            
            # 合并特征
            all_features = {
                **f0_features,
                **energy_features,
                **pause_features,
                **spectral_features
            }
            
            if include_mfcc:
                mfcc_features = self.extract_mfcc_features(audio)
                all_features.update(mfcc_features)
            
            # 转换为数组
            feature_vector = np.array(list(all_features.values()), dtype=np.float32)
            
            # 处理NaN值
            feature_vector = np.nan_to_num(feature_vector, nan=0.0)
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"特征提取失败 {audio_path}: {e}")
            # 返回零向量
            n_features = 18 if not include_mfcc else 44  # 18基础特征 + 26 MFCC
            return np.zeros(n_features, dtype=np.float32)
    
    def batch_extract(
        self, 
        audio_paths: List[str],
        include_mfcc: bool = False
    ) -> np.ndarray:
        """批量提取特征"""
        
        features = []
        for audio_path in audio_paths:
            feature_vector = self.extract_all_features(audio_path, include_mfcc)
            features.append(feature_vector)
            
        return np.array(features)
    
    def get_feature_names(self, include_mfcc: bool = False) -> List[str]:
        """获取特征名称"""
        
        base_features = [
            # F0特征
            'f0_mean', 'f0_std', 'f0_range', 'f0_median',
            # 能量特征
            'energy_mean', 'energy_std', 'energy_max', 'energy_dynamic_range',
            # 停顿特征
            'pause_count', 'pause_total_time', 'pause_ratio', 
            'pause_mean_duration', 'pause_frequency',
            # 谱特征
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_bandwidth_mean', 'spectral_contrast_mean',
            'spectral_rolloff_mean'
        ]
        
        if include_mfcc:
            mfcc_features = []
            for i in range(13):
                mfcc_features.extend([f'mfcc_{i}_mean', f'mfcc_{i}_std'])
            base_features.extend(mfcc_features)
            
        return base_features

