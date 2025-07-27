import torch
import torchaudio
import librosa
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class AudioProcessor:
    """音频处理器
    
    负责音频信号的加载、预处理和特征提取
    """
    
    def __init__(self, audio_config: Dict):
        """
        Args:
            audio_config: 音频处理配置
        """
        self.config = audio_config
        self.sample_rate = audio_config['sample_rate']
        self.max_duration = audio_config['max_duration']
        self.min_duration = audio_config['min_duration']
        self.normalize = audio_config['normalize']
        self.remove_silence = audio_config['remove_silence']
        self.silence_threshold = audio_config['silence_threshold']
        
        logger.info(f"初始化音频处理器: sr={self.sample_rate}, max_dur={self.max_duration}s")
    
    def process(self, audio_path: str) -> torch.Tensor:
        """处理音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频特征张量 [T, D]
        """
        try:
            # 加载音频
            waveform, sr = torchaudio.load(audio_path)
            
            # 重采样
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # 单声道转换
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 移除静音
            if self.remove_silence:
                waveform = self._remove_silence(waveform)
            
            # 长度调整
            waveform = self._adjust_length(waveform)
            
            # 归一化
            if self.normalize:
                waveform = self._normalize(waveform)
            
            # 提取特征 (这里使用简单的MFCC特征，实际应用中会使用预训练模型)
            features = self._extract_features(waveform.squeeze(0))
            
            return features
            
        except Exception as e:
            logger.error(f"处理音频文件 {audio_path} 时出错: {e}")
            # 返回零填充的特征
            return torch.zeros(1000, 39)  # 39维MFCC特征
    
    def _remove_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        """移除静音片段"""
        try:
            # 使用librosa的静音检测
            audio_np = waveform.squeeze().numpy()
            non_silent_intervals = librosa.effects.split(
                audio_np, 
                top_db=-self.silence_threshold,
                frame_length=2048,
                hop_length=512
            )
            
            if len(non_silent_intervals) == 0:
                return waveform
            
            # 连接非静音片段
            non_silent_audio = []
            for start, end in non_silent_intervals:
                non_silent_audio.append(audio_np[start:end])
            
            if non_silent_audio:
                cleaned_audio = np.concatenate(non_silent_audio)
                return torch.tensor(cleaned_audio).unsqueeze(0)
            else:
                return waveform
                
        except Exception as e:
            logger.warning(f"静音移除失败: {e}")
            return waveform
    
    def _adjust_length(self, waveform: torch.Tensor) -> torch.Tensor:
        """调整音频长度"""
        current_length = waveform.shape[1]
        target_length = int(self.max_duration * self.sample_rate)
        
        if current_length > target_length:
            # 截断
            waveform = waveform[:, :target_length]
        elif current_length < target_length:
            # 填充
            padding = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform
    
    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """音频归一化"""
        # RMS归一化
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > 0:
            waveform = waveform / rms
        
        # 峰值归一化
        max_val = torch.max(torch.abs(waveform))
        if max_val > 1.0:
            waveform = waveform / max_val
            
        return waveform
    
    def _extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """提取音频特征
        
        Args:
            waveform: 音频波形 [T]
            
        Returns:
            特征张量 [T_feat, D]
        """
        try:
            # 转换为numpy用于librosa处理
            audio_np = waveform.numpy()
            
            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(
                y=audio_np,
                sr=self.sample_rate,
                n_mfcc=13,
                hop_length=512,
                n_fft=2048
            )
            
            # 提取MFCC一阶和二阶差分
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # 合并特征
            features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
            
            # 转置为 [T, D] 格式
            features = features.T
            
            return torch.tensor(features, dtype=torch.float32)
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            # 返回零填充的特征
            n_frames = waveform.shape[0] // 512  # 根据hop_length计算
            return torch.zeros(max(n_frames, 1), 39)
    
    def extract_pause_features(self, waveform: torch.Tensor) -> Dict[str, float]:
        """提取停顿相关特征
        
        Args:
            waveform: 音频波形
            
        Returns:
            停顿特征字典
        """
        try:
            audio_np = waveform.squeeze().numpy()
            
            # 计算RMS能量
            hop_length = 512
            frame_length = 2048
            rms = librosa.feature.rms(
                y=audio_np, 
                frame_length=frame_length,
                hop_length=hop_length
            )[0]
            
            # 静音检测阈值
            silence_threshold = np.percentile(rms, 20)  # 20%分位数作为阈值
            
            # 检测静音帧
            silence_frames = rms < silence_threshold
            
            # 计算停顿统计
            total_frames = len(rms)
            silence_frame_count = np.sum(silence_frames)
            pause_ratio = silence_frame_count / total_frames if total_frames > 0 else 0
            
            # 检测停顿段
            pause_segments = self._detect_pause_segments(silence_frames, hop_length)
            
            features = {
                'pause_ratio': float(pause_ratio),
                'num_pauses': len(pause_segments),
                'avg_pause_duration': float(np.mean([seg['duration'] for seg in pause_segments])) if pause_segments else 0.0,
                'total_pause_time': float(sum([seg['duration'] for seg in pause_segments])),
                'speech_time': float((total_frames - silence_frame_count) * hop_length / self.sample_rate)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"停顿特征提取失败: {e}")
            return {
                'pause_ratio': 0.0,
                'num_pauses': 0,
                'avg_pause_duration': 0.0,
                'total_pause_time': 0.0,
                'speech_time': 0.0
            }
    
    def _detect_pause_segments(self, silence_frames: np.ndarray, hop_length: int) -> list:
        """检测停顿段"""
        pause_segments = []
        in_pause = False
        pause_start = 0
        
        for i, is_silent in enumerate(silence_frames):
            if is_silent and not in_pause:
                # 开始停顿
                in_pause = True
                pause_start = i
            elif not is_silent and in_pause:
                # 结束停顿
                in_pause = False
                duration = (i - pause_start) * hop_length / self.sample_rate
                if duration > 0.1:  # 只记录超过100ms的停顿
                    pause_segments.append({
                        'start': pause_start * hop_length / self.sample_rate,
                        'end': i * hop_length / self.sample_rate,
                        'duration': duration
                    })
        
        return pause_segments
    
    def calculate_speech_rate(self, waveform: torch.Tensor, text: Optional[str] = None) -> float:
        """计算语速
        
        Args:
            waveform: 音频波形
            text: 转录文本(可选)
            
        Returns:
            语速 (词/秒)
        """
        try:
            # 计算有效语音时间
            pause_features = self.extract_pause_features(waveform)
            speech_time = pause_features['speech_time']
            
            if text and speech_time > 0:
                # 基于文本计算词数
                word_count = len(text.split())
                speech_rate = word_count / speech_time
            else:
                # 基于音频特征估算语速
                # 这里使用一个简化的方法，实际应用中可能需要更复杂的算法
                audio_np = waveform.squeeze().numpy()
                
                # 计算过零率
                zcr = librosa.feature.zero_crossing_rate(audio_np, hop_length=512)[0]
                avg_zcr = np.mean(zcr)
                
                # 基于过零率估算语速 (这是一个很粗糙的估算)
                speech_rate = avg_zcr * 10  # 经验公式
            
            return float(speech_rate)
            
        except Exception as e:
            logger.error(f"语速计算失败: {e}")
            return 0.0 