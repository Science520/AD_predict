import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Wav2Vec2Model, Wav2Vec2Processor,
    WhisperModel, WhisperProcessor,
    AutoModel, AutoProcessor
)
from typing import Dict, Optional, Tuple, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)


class TransformerASR(nn.Module):
    """基于Transformer的自动语音识别模型
    
    支持Wav2Vec2和Whisper等预训练模型
    """
    
    def __init__(self, asr_config: Dict):
        super().__init__()
        self.config = asr_config
        self.model_name = asr_config['model_name']
        self.sample_rate = asr_config['sample_rate']
        self.max_length = asr_config['max_length']
        self.feature_dim = asr_config['feature_dim']
        self.pause_threshold_ms = asr_config.get('pause_threshold_ms', 250)
        
        # 初始化预训练模型
        self._initialize_model()
        
        # 特征提取层
        self.feature_projector = nn.Linear(
            self.pretrained_feature_dim, 
            self.feature_dim
        )
        
        # 停顿检测器
        self.pause_detector = PauseDetector(
            input_dim=self.feature_dim,
            threshold_ms=self.pause_threshold_ms,
            sample_rate=self.sample_rate
        )
        
        logger.info(f"初始化ASR模型: {self.model_name}")
    
    def _initialize_model(self):
        """初始化预训练模型"""
        try:
            if 'wav2vec2' in self.model_name.lower():
                self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
                self.model = Wav2Vec2Model.from_pretrained(self.model_name)
                self.pretrained_feature_dim = self.model.config.hidden_size
                self.model_type = 'wav2vec2'
                
            elif 'whisper' in self.model_name.lower():
                self.processor = WhisperProcessor.from_pretrained(self.model_name)
                self.model = WhisperModel.from_pretrained(self.model_name)
                self.pretrained_feature_dim = self.model.config.d_model
                self.model_type = 'whisper'
                
            else:
                # 尝试使用AutoModel
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.pretrained_feature_dim = self.model.config.hidden_size
                self.model_type = 'auto'
            
            # 冻结预训练模型参数 (可选)
            if not self.config.get('fine_tune_pretrained', True):
                for param in self.model.parameters():
                    param.requires_grad = False
                    
        except Exception as e:
            logger.error(f"无法加载预训练模型 {self.model_name}: {e}")
            raise e
    
    def forward(
        self, 
        audio_input: Union[torch.Tensor, str],
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            audio_input: 音频输入 [B, T] 或音频文件路径
            return_attention: 是否返回注意力权重
            
        Returns:
            Dict包含:
            - acoustic_features: 声学特征 [B, T, D]
            - text_features: 文本特征 [B, D] (如果支持)
            - pause_predictions: 停顿预测 [B, T]
            - attention_weights: 注意力权重 (可选)
        """
        # 预处理音频
        if isinstance(audio_input, str):
            # 如果是文件路径，加载音频
            processed_audio = self._process_audio_file(audio_input)
        else:
            processed_audio = audio_input
        
        # 提取特征
        with torch.no_grad() if not self.config.get('fine_tune_pretrained', True) else torch.enable_grad():
            pretrained_features = self._extract_pretrained_features(processed_audio)
        
        # 投影到目标维度
        acoustic_features = self.feature_projector(pretrained_features)
        
        # 停顿检测
        pause_predictions = self.pause_detector(acoustic_features)
        
        # 文本特征 (如果支持转录)
        text_features = self._extract_text_features(acoustic_features)
        
        outputs = {
            'acoustic_features': acoustic_features,
            'text_features': text_features,
            'pause_predictions': pause_predictions
        }
        
        if return_attention:
            outputs['attention_weights'] = self._get_attention_weights()
        
        return outputs
    
    def _process_audio_file(self, audio_path: str) -> torch.Tensor:
        """处理音频文件"""
        try:
            import torchaudio
            waveform, sr = torchaudio.load(audio_path)
            
            # 重采样到目标采样率
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # 单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            return waveform.squeeze(0)  # [T]
            
        except Exception as e:
            logger.error(f"音频文件处理失败: {e}")
            # 返回零填充
            return torch.zeros(self.sample_rate * self.max_length)
    
    def _extract_pretrained_features(self, audio: torch.Tensor) -> torch.Tensor:
        """使用预训练模型提取特征"""
        try:
            if self.model_type == 'wav2vec2':
                # Wav2Vec2处理
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)  # [1, T]
                
                # 使用processor预处理
                inputs = self.processor(
                    audio.cpu().numpy(), 
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    padding=True
                )
                
                # 移动到正确设备
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(audio.device)
                
                # 提取特征
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state  # [B, T, D]
                
            elif self.model_type == 'whisper':
                # Whisper处理
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                
                # Whisper需要log-mel spectrogram
                inputs = self.processor(
                    audio.cpu().numpy(),
                    sampling_rate=self.sample_rate, 
                    return_tensors="pt"
                )
                
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(audio.device)
                
                # 只使用encoder
                encoder_outputs = self.model.encoder(**inputs)
                features = encoder_outputs.last_hidden_state  # [B, T, D]
                
            else:
                # 通用处理
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                
                inputs = self.processor(
                    audio.cpu().numpy(),
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    padding=True
                )
                
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(audio.device)
                
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state
            
            return features
            
        except Exception as e:
            logger.error(f"预训练特征提取失败: {e}")
            # 返回随机特征
            batch_size = 1 if audio.dim() == 1 else audio.shape[0]
            seq_len = min(1000, audio.shape[-1] // 320)  # 估算序列长度
            return torch.randn(batch_size, seq_len, self.pretrained_feature_dim, device=audio.device)
    
    def _extract_text_features(self, acoustic_features: torch.Tensor) -> torch.Tensor:
        """从声学特征中提取文本级别特征"""
        # 使用全局平均池化获得句子级表示
        text_features = torch.mean(acoustic_features, dim=1)  # [B, D]
        
        return text_features
    
    def _get_attention_weights(self) -> Optional[torch.Tensor]:
        """获取注意力权重 (如果模型支持)"""
        # 这里需要根据具体模型实现
        return None
    
    def transcribe(self, audio_input: Union[torch.Tensor, str]) -> str:
        """转录音频为文本
        
        Args:
            audio_input: 音频输入
            
        Returns:
            转录文本
        """
        try:
            if self.model_type == 'whisper':
                # Whisper支持端到端转录
                if isinstance(audio_input, str):
                    audio = self._process_audio_file(audio_input)
                else:
                    audio = audio_input
                
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                
                # 使用Whisper进行转录
                from transformers import WhisperForConditionalGeneration, WhisperTokenizer
                
                # 这里需要加载完整的Whisper模型
                full_model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
                tokenizer = WhisperTokenizer.from_pretrained(self.model_name)
                
                inputs = self.processor(
                    audio.cpu().numpy(),
                    sampling_rate=self.sample_rate,
                    return_tensors="pt"
                )
                
                with torch.no_grad():
                    generated_ids = full_model.generate(**inputs)
                
                transcription = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                return transcription
            else:
                # 对于Wav2Vec2等模型，需要额外的解码器
                logger.warning("当前模型不支持直接转录，返回占位符文本")
                return "Transcription not available for this model type"
                
        except Exception as e:
            logger.error(f"转录失败: {e}")
            return "Transcription failed"
    
    def detect_pauses(self, audio_input: Union[torch.Tensor, str]) -> Dict[str, Union[torch.Tensor, list]]:
        """检测语音中的停顿
        
        Args:
            audio_input: 音频输入
            
        Returns:
            停顿检测结果
        """
        outputs = self.forward(audio_input)
        pause_predictions = outputs['pause_predictions']
        
        # 后处理：找到停顿段
        pause_segments = self._postprocess_pauses(pause_predictions)
        
        return {
            'pause_predictions': pause_predictions,
            'pause_segments': pause_segments,
            'pause_ratio': torch.mean(pause_predictions.float()),
            'num_pauses': len(pause_segments)
        }
    
    def _postprocess_pauses(self, pause_predictions: torch.Tensor) -> list:
        """后处理停顿预测结果"""
        # 将概率转换为二值预测
        binary_pauses = (pause_predictions > 0.5).float()
        
        # 找到连续的停顿段
        pause_segments = []
        in_pause = False
        start_idx = 0
        
        for i, is_pause in enumerate(binary_pauses.squeeze()):
            if is_pause and not in_pause:
                in_pause = True
                start_idx = i
            elif not is_pause and in_pause:
                in_pause = False
                # 计算时间 (假设每个时间步对应某个固定时间)
                time_per_step = 0.02  # 20ms per step
                start_time = start_idx * time_per_step
                end_time = i * time_per_step
                
                if end_time - start_time > self.pause_threshold_ms / 1000:  # 转换为秒
                    pause_segments.append({
                        'start': start_time,
                        'end': end_time,
                        'duration': end_time - start_time
                    })
        
        return pause_segments


class PauseDetector(nn.Module):
    """停顿检测器"""
    
    def __init__(self, input_dim: int, threshold_ms: int = 250, sample_rate: int = 16000):
        super().__init__()
        self.threshold_ms = threshold_ms
        self.sample_rate = sample_rate
        
        # 停顿检测网络
        self.detector = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 输入特征 [B, T, D]
            
        Returns:
            pause_probs: 停顿概率 [B, T, 1]
        """
        batch_size, seq_len, feature_dim = features.shape
        
        # Reshape for processing
        features_flat = features.view(-1, feature_dim)  # [B*T, D]
        
        # 预测停顿概率
        pause_probs = self.detector(features_flat)  # [B*T, 1]
        
        # Reshape back
        pause_probs = pause_probs.view(batch_size, seq_len, 1)  # [B, T, 1]
        
        return pause_probs 