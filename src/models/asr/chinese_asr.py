#!/usr/bin/env python3
"""
中文ASR模型 - 支持老年人语音识别和停顿检测
"""
import torch
import torch.nn as nn
import torchaudio
import whisper
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import librosa
import soundfile as sf
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ASROutput:
    """ASR输出结果"""
    text: str
    segments: List[Dict]  # 包含时间戳的分段
    pause_info: Dict  # 停顿信息
    acoustic_features: torch.Tensor  # 声学特征
    confidence_scores: List[float]  # 置信度分数

class ChineseASR(nn.Module):
    """中文ASR模型 - 基于Whisper，针对老年人语音优化"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 加载Whisper模型
        model_name = config.get('model_name', 'openai/whisper-large-v3')
        if 'whisper' in model_name.lower():
            self.model = whisper.load_model(model_name.split('/')[-1])
        else:
            raise ValueError(f"不支持的模型: {model_name}")
            
        self.sample_rate = config.get('sample_rate', 16000)
        self.language = config.get('language', 'zh')
        self.pause_threshold = config.get('pause_threshold_ms', 500) / 1000  # 转为秒
        self.min_pause_duration = config.get('min_pause_duration', 0.3)
        
        # 老年人语音特定配置
        elderly_config = config.get('elderly_specific', {})
        self.speech_rate_adjustment = elderly_config.get('speech_rate_adjustment', 0.8)
        self.silence_tolerance = elderly_config.get('silence_tolerance', 1.0)
        self.accent_adaptation = elderly_config.get('accent_adaptation', True)
        
        # VAD (语音活动检测)
        self.vad_threshold = config.get('vad_threshold', 0.5)
        
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """音频预处理 - 针对老年人语音优化"""
        
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # 老年人语音增强
        if self.accent_adaptation:
            # 语音增强：降噪
            audio = librosa.effects.preemphasis(audio)
            
            # 动态范围压缩（老年人语音音量可能不稳定）
            audio = librosa.util.normalize(audio)
            
        return audio
    
    def detect_pauses(self, audio: np.ndarray, segments: List[Dict]) -> Dict:
        """检测停顿信息"""
        
        # 计算音频能量
        frame_length = int(0.025 * self.sample_rate)  # 25ms帧
        hop_length = int(0.01 * self.sample_rate)     # 10ms步长
        
        energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # 检测静音段
        silence_mask = energy < (np.mean(energy) * 0.1)
        
        # 转换为时间戳
        time_frames = librosa.frames_to_time(
            np.arange(len(energy)),
            sr=self.sample_rate,
            hop_length=hop_length
        )
        
        # 找到停顿段落
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
                    pauses.append({
                        'start': pause_start,
                        'end': time_frames[i],
                        'duration': pause_duration
                    })
                in_pause = False
        
        # 计算停顿统计
        total_pause_time = sum(p['duration'] for p in pauses)
        audio_duration = len(audio) / self.sample_rate
        pause_ratio = total_pause_time / audio_duration if audio_duration > 0 else 0
        
        return {
            'pauses': pauses,
            'total_pause_time': total_pause_time,
            'pause_ratio': pause_ratio,
            'pause_count': len(pauses),
            'average_pause_duration': total_pause_time / len(pauses) if pauses else 0
        }
    
    def extract_acoustic_features(self, audio: np.ndarray) -> torch.Tensor:
        """提取声学特征"""
        
        # 使用Whisper编码器提取特征
        audio_tensor = torch.from_numpy(audio).float()
        
        # 填充到Whisper要求的长度
        if len(audio_tensor) < self.sample_rate * 30:
            padding = self.sample_rate * 30 - len(audio_tensor)
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
        else:
            audio_tensor = audio_tensor[:self.sample_rate * 30]
            
        # 通过Whisper编码器
        with torch.no_grad():
            mel = whisper.log_mel_spectrogram(audio_tensor)
            features = self.model.encoder(mel.unsqueeze(0))
            
        return features.squeeze(0)  # [seq_len, dim]
    
    def calculate_speech_rate(self, text: str, audio_duration: float, pause_info: Dict) -> float:
        """计算语速 (词/分钟)"""
        
        # 简单的中文分词 (按字符计算)
        word_count = len([c for c in text if c.strip() and not c.isspace()])
        
        # 减去停顿时间
        effective_duration = audio_duration - pause_info['total_pause_time']
        
        if effective_duration > 0:
            speech_rate = (word_count / effective_duration) * 60  # 字/分钟
        else:
            speech_rate = 0
            
        return speech_rate
    
    def add_pause_markers(self, segments: List[Dict], pause_info: Dict) -> str:
        """在文本中添加停顿标记"""
        
        # 将所有分段按时间排序
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        pauses = pause_info['pauses']
        
        result_text = ""
        last_end = 0
        
        for segment in sorted_segments:
            seg_start = segment['start']
            seg_text = segment['text']
            
            # 检查是否在此分段前有停顿
            for pause in pauses:
                if last_end <= pause['start'] <= seg_start:
                    if pause['duration'] >= self.pause_threshold:
                        result_text += f" <pause:{pause['duration']:.1f}s> "
                        
            result_text += seg_text
            last_end = segment['end']
            
        return result_text.strip()
    
    def forward(self, audio_path: str) -> ASROutput:
        """前向传播 - 完整的ASR处理流程"""
        
        try:
            # 1. 音频预处理
            audio = self.preprocess_audio(audio_path)
            audio_duration = len(audio) / self.sample_rate
            
            # 2. Whisper转录
            result = self.model.transcribe(
                audio,
                language=self.language,
                task='transcribe',
                word_timestamps=True,
                verbose=False
            )
            
            text = result['text']
            segments = result.get('segments', [])
            
            # 3. 停顿检测
            pause_info = self.detect_pauses(audio, segments)
            
            # 4. 添加停顿标记到文本
            text_with_pauses = self.add_pause_markers(segments, pause_info)
            
            # 5. 提取声学特征
            acoustic_features = self.extract_acoustic_features(audio)
            
            # 6. 计算语速
            speech_rate = self.calculate_speech_rate(text, audio_duration, pause_info)
            
            # 7. 置信度分数
            confidence_scores = [seg.get('avg_logprob', 0.0) for seg in segments]
            
            # 8. 汇总结果
            enhanced_pause_info = {
                **pause_info,
                'speech_rate': speech_rate,
                'audio_duration': audio_duration
            }
            
            return ASROutput(
                text=text_with_pauses,
                segments=segments,
                pause_info=enhanced_pause_info,
                acoustic_features=acoustic_features,
                confidence_scores=confidence_scores
            )
            
        except Exception as e:
            logger.error(f"ASR处理失败: {e}")
            # 返回空结果
            return ASROutput(
                text="",
                segments=[],
                pause_info={},
                acoustic_features=torch.zeros(1, self.config['feature_dim']),
                confidence_scores=[]
            )
    
    def batch_process(self, audio_paths: List[str]) -> List[ASROutput]:
        """批量处理音频文件"""
        
        results = []
        for audio_path in audio_paths:
            result = self.forward(audio_path)
            results.append(result)
            
        return results

def create_chinese_asr_model(config: Dict) -> ChineseASR:
    """创建中文ASR模型实例"""
    
    model = ChineseASR(config)
    
    logger.info(f"创建中文ASR模型: {config.get('model_name')}")
    logger.info(f"支持语言: {config.get('language', 'zh')}")
    logger.info(f"老年人语音优化: {config.get('elderly_specific', {})}")
    
    return model 