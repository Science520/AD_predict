#!/usr/bin/env python3
"""
增强停顿检测器 - 专门针对老年人语音优化
结合传统信号处理和深度学习方法
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PauseSegment:
    """停顿片段"""
    start: float
    end: float
    duration: float
    confidence: float
    pause_type: str  # 'short', 'medium', 'long'

class ElderlyPauseDetector(nn.Module):
    """老年人语音停顿检测器 - 混合方法"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.sample_rate = config.get('sample_rate', 16000)
        
        # 老年人语音特定参数
        self.elderly_config = config.get('elderly_specific', {})
        self.slow_speech_factor = self.elderly_config.get('slow_speech_factor', 1.5)
        self.energy_threshold_factor = self.elderly_config.get('energy_threshold_factor', 0.05)
        self.min_pause_duration = self.elderly_config.get('min_pause_duration', 0.2)  # 更短的最小停顿
        
        # 停顿分类阈值（秒）
        self.short_pause_threshold = 0.3
        self.medium_pause_threshold = 0.8
        self.long_pause_threshold = 1.5
        
        # 深度学习停顿检测器
        feature_dim = config.get('feature_dim', 768)
        self.neural_detector = self._build_neural_detector(feature_dim)
        
        # 加权融合参数
        self.energy_weight = config.get('energy_weight', 0.4)
        self.neural_weight = config.get('neural_weight', 0.6)
        
    def _build_neural_detector(self, input_dim: int) -> nn.Module:
        """构建神经网络停顿检测器"""
        return nn.Sequential(
            # 时间卷积层 - 捕捉时序模式
            nn.Conv1d(input_dim, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 双向LSTM - 捕捉上下文
            nn.LSTM(128, 64, bidirectional=True, batch_first=True),
            
            # 输出层
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def detect_pauses_energy_based(self, audio: np.ndarray) -> List[PauseSegment]:
        """基于能量的停顿检测 - 针对老年人语音优化"""
        
        # 计算短时能量
        frame_length = int(0.025 * self.sample_rate)  # 25ms
        hop_length = int(0.01 * self.sample_rate)     # 10ms
        
        # 使用多种能量特征
        rms_energy = librosa.feature.rms(
            y=audio, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        # 谱质心 - 帮助区分语音和噪声
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, 
            sr=self.sample_rate,
            hop_length=hop_length
        )[0]
        
        # 零交叉率 - 检测静音段
        zcr = librosa.feature.zero_crossing_rate(
            y=audio, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        # 自适应阈值计算
        # 对于老年人语音，使用更低的能量阈值
        median_energy = np.median(rms_energy)
        energy_threshold = median_energy * self.energy_threshold_factor
        
        # 谱质心阈值（语音通常有更高的谱质心）
        centroid_threshold = np.median(spectral_centroids) * 0.5
        
        # 综合判断静音段
        silence_mask = (
            (rms_energy < energy_threshold) |
            (spectral_centroids < centroid_threshold) |
            (zcr > 0.5)  # 高零交叉率通常表示噪声或静音
        )
        
        # 形态学处理 - 填补短暂的间隙
        from scipy import ndimage
        silence_mask = ndimage.binary_opening(silence_mask, iterations=2)
        silence_mask = ndimage.binary_closing(silence_mask, iterations=3)
        
        # 转换为时间戳
        time_frames = librosa.frames_to_time(
            np.arange(len(silence_mask)),
            sr=self.sample_rate,
            hop_length=hop_length
        )
        
        # 找到停顿段
        pauses = []
        in_pause = False
        pause_start = 0
        
        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_pause:
                pause_start = time_frames[i]
                in_pause = True
            elif not is_silent and in_pause:
                pause_duration = time_frames[i] - pause_start
                if pause_duration >= self.min_pause_duration:
                    # 计算该停顿段的置信度
                    start_idx = int(pause_start * self.sample_rate / hop_length)
                    end_idx = int(time_frames[i] * self.sample_rate / hop_length)
                    segment_energy = rms_energy[start_idx:end_idx]
                    confidence = 1.0 - np.mean(segment_energy) / (median_energy + 1e-8)
                    
                    # 停顿类型分类
                    if pause_duration < self.short_pause_threshold:
                        pause_type = 'short'
                    elif pause_duration < self.medium_pause_threshold:
                        pause_type = 'medium'
                    else:
                        pause_type = 'long'
                    
                    pauses.append(PauseSegment(
                        start=pause_start,
                        end=time_frames[i],
                        duration=pause_duration,
                        confidence=min(confidence, 1.0),
                        pause_type=pause_type
                    ))
                in_pause = False
        
        return pauses
    
    def detect_pauses_neural(self, features: torch.Tensor) -> torch.Tensor:
        """基于神经网络的停顿检测"""
        
        if features.dim() == 3:  # [B, T, D]
            batch_size, seq_len, feature_dim = features.shape
            
            # 转置为卷积所需格式 [B, D, T]
            features_conv = features.transpose(1, 2)
            
            # 通过卷积层
            conv_out = features_conv
            for layer in self.neural_detector[:6]:  # 前6层是卷积和归一化
                if isinstance(layer, nn.Conv1d):
                    conv_out = layer(conv_out)
                else:
                    conv_out = layer(conv_out)
            
            # 转回LSTM格式 [B, T, D]
            lstm_input = conv_out.transpose(1, 2)
            
            # 通过LSTM
            lstm_out, _ = self.neural_detector[6](lstm_input)
            
            # 通过最终的线性层
            output = lstm_out.view(-1, lstm_out.size(-1))
            for layer in self.neural_detector[7:]:
                output = layer(output)
            
            # 重新整形
            pause_probs = output.view(batch_size, seq_len, 1)
            
        else:
            # 处理2D输入
            pause_probs = self.neural_detector(features)
        
        return pause_probs
    
    def fuse_detections(
        self, 
        energy_pauses: List[PauseSegment], 
        neural_probs: torch.Tensor,
        audio_duration: float
    ) -> List[PauseSegment]:
        """融合两种检测方法的结果"""
        
        # 将神经网络输出转换为停顿段
        neural_pauses = self._neural_probs_to_segments(neural_probs, audio_duration)
        
        # 融合策略：使用加权投票
        fused_pauses = []
        
        # 创建时间网格
        time_resolution = 0.01  # 10ms分辨率
        max_time = max(audio_duration, 
                      max([p.end for p in energy_pauses] + [0]),
                      max([p.end for p in neural_pauses] + [0]))
        
        time_grid = np.arange(0, max_time, time_resolution)
        energy_votes = np.zeros(len(time_grid))
        neural_votes = np.zeros(len(time_grid))
        
        # 能量检测投票
        for pause in energy_pauses:
            start_idx = int(pause.start / time_resolution)
            end_idx = int(pause.end / time_resolution)
            if end_idx < len(energy_votes):
                energy_votes[start_idx:end_idx] = pause.confidence
        
        # 神经网络检测投票
        for pause in neural_pauses:
            start_idx = int(pause.start / time_resolution)
            end_idx = int(pause.end / time_resolution)
            if end_idx < len(neural_votes):
                neural_votes[start_idx:end_idx] = pause.confidence
        
        # 加权融合
        fused_votes = (self.energy_weight * energy_votes + 
                      self.neural_weight * neural_votes)
        
        # 阈值化和连通分析
        threshold = 0.5
        pause_mask = fused_votes > threshold
        
        # 找到连续的停顿段
        in_pause = False
        pause_start_idx = 0
        
        for i, is_pause in enumerate(pause_mask):
            if is_pause and not in_pause:
                pause_start_idx = i
                in_pause = True
            elif not is_pause and in_pause:
                pause_duration = (i - pause_start_idx) * time_resolution
                if pause_duration >= self.min_pause_duration:
                    start_time = pause_start_idx * time_resolution
                    end_time = i * time_resolution
                    
                    # 计算融合置信度
                    segment_confidence = np.mean(fused_votes[pause_start_idx:i])
                    
                    # 停顿类型
                    if pause_duration < self.short_pause_threshold:
                        pause_type = 'short'
                    elif pause_duration < self.medium_pause_threshold:
                        pause_type = 'medium'
                    else:
                        pause_type = 'long'
                    
                    fused_pauses.append(PauseSegment(
                        start=start_time,
                        end=end_time,
                        duration=pause_duration,
                        confidence=segment_confidence,
                        pause_type=pause_type
                    ))
                in_pause = False
        
        return fused_pauses
    
    def _neural_probs_to_segments(
        self, 
        neural_probs: torch.Tensor, 
        audio_duration: float
    ) -> List[PauseSegment]:
        """将神经网络概率转换为停顿段"""
        
        if neural_probs.dim() == 3:
            neural_probs = neural_probs.squeeze(-1)  # [B, T]
        if neural_probs.dim() == 2:
            neural_probs = neural_probs[0]  # 取第一个batch
        
        # 时间分辨率
        time_per_frame = audio_duration / len(neural_probs)
        
        # 阈值化
        threshold = 0.5
        pause_mask = neural_probs > threshold
        
        # 找停顿段
        pauses = []
        in_pause = False
        pause_start_idx = 0
        
        for i, is_pause in enumerate(pause_mask):
            if is_pause and not in_pause:
                pause_start_idx = i
                in_pause = True
            elif not is_pause and in_pause:
                pause_duration = (i - pause_start_idx) * time_per_frame
                if pause_duration >= self.min_pause_duration:
                    start_time = pause_start_idx * time_per_frame
                    end_time = i * time_per_frame
                    confidence = torch.mean(neural_probs[pause_start_idx:i]).item()
                    
                    # 停顿类型
                    if pause_duration < self.short_pause_threshold:
                        pause_type = 'short'
                    elif pause_duration < self.medium_pause_threshold:
                        pause_type = 'medium'
                    else:
                        pause_type = 'long'
                    
                    pauses.append(PauseSegment(
                        start=start_time,
                        end=end_time,
                        duration=pause_duration,
                        confidence=confidence,
                        pause_type=pause_type
                    ))
                in_pause = False
        
        return pauses
    
    def forward(
        self, 
        audio: np.ndarray, 
        features: Optional[torch.Tensor] = None
    ) -> Dict:
        """完整的停顿检测流程"""
        
        audio_duration = len(audio) / self.sample_rate
        
        # 1. 基于能量的检测
        energy_pauses = self.detect_pauses_energy_based(audio)
        
        # 2. 基于神经网络的检测
        if features is not None:
            neural_probs = self.detect_pauses_neural(features)
        else:
            # 如果没有提供特征，创建假特征用于演示
            seq_len = int(audio_duration * 100)  # 假设100Hz特征
            features = torch.randn(1, seq_len, self.config.get('feature_dim', 768))
            neural_probs = self.detect_pauses_neural(features)
        
        # 3. 融合两种方法
        fused_pauses = self.fuse_detections(energy_pauses, neural_probs, audio_duration)
        
        # 4. 计算统计信息
        total_pause_time = sum(p.duration for p in fused_pauses)
        pause_ratio = total_pause_time / audio_duration if audio_duration > 0 else 0
        
        # 按类型统计
        pause_counts = {'short': 0, 'medium': 0, 'long': 0}
        for pause in fused_pauses:
            pause_counts[pause.pause_type] += 1
        
        return {
            'pauses': fused_pauses,
            'energy_pauses': energy_pauses,
            'total_pause_time': total_pause_time,
            'pause_ratio': pause_ratio,
            'pause_count': len(fused_pauses),
            'pause_counts_by_type': pause_counts,
            'average_pause_duration': total_pause_time / len(fused_pauses) if fused_pauses else 0,
            'elderly_specific_metrics': self._calculate_elderly_metrics(fused_pauses, audio_duration)
        }
    
    def _calculate_elderly_metrics(self, pauses: List[PauseSegment], audio_duration: float) -> Dict:
        """计算老年人语音特定的指标"""
        
        if not pauses:
            return {
                'speech_continuity': 1.0,
                'pause_frequency': 0.0,
                'long_pause_ratio': 0.0
            }
        
        # 语音连续性：连续语音段的平均长度
        speech_segments = []
        last_end = 0
        for pause in sorted(pauses, key=lambda x: x.start):
            if pause.start > last_end:
                speech_segments.append(pause.start - last_end)
            last_end = pause.end
        
        if audio_duration > last_end:
            speech_segments.append(audio_duration - last_end)
        
        avg_speech_segment = np.mean(speech_segments) if speech_segments else 0
        
        # 停顿频率：每分钟停顿次数
        pause_frequency = len(pauses) / (audio_duration / 60) if audio_duration > 0 else 0
        
        # 长停顿比例
        long_pauses = [p for p in pauses if p.pause_type == 'long']
        long_pause_ratio = len(long_pauses) / len(pauses) if pauses else 0
        
        return {
            'speech_continuity': avg_speech_segment,
            'pause_frequency': pause_frequency,
            'long_pause_ratio': long_pause_ratio
        }

def create_elderly_pause_detector(config: Dict) -> ElderlyPauseDetector:
    """创建老年人停顿检测器"""
    
    detector = ElderlyPauseDetector(config)
    
    logger.info("创建老年人停顿检测器")
    logger.info(f"最小停顿时长: {detector.min_pause_duration}s")
    logger.info(f"能量权重: {detector.energy_weight}, 神经网络权重: {detector.neural_weight}")
    
    return detector 