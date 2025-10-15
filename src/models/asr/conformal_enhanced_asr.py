#!/usr/bin/env python3
"""
增强版ASR - 集成Conformal Inference

结合原有的中文ASR和Conformal预测，提供带不确定性量化的转录
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import librosa

from .chinese_asr import ChineseASR, ASROutput
from ..conformal.conformal_asr import ConformalASR, ConformalPredictionSet

logger = logging.getLogger(__name__)


@dataclass
class ConformalASROutput:
    """带Conformal预测的ASR输出"""
    # 标准ASR输出
    text: str
    segments: List[Dict]
    pause_info: Dict
    acoustic_features: torch.Tensor
    confidence_scores: List[float]
    
    # Conformal预测
    prediction_set: List[str]  # 候选转录集合
    set_size: int  # 预测集大小
    conformal_confidence: float  # Conformal置信度
    coverage_probability: float  # 覆盖概率


class ConformalEnhancedASR:
    """
    集成Conformal Inference的增强ASR
    
    结合了：
    1. 中文ASR的停顿检测和声学特征提取
    2. Conformal预测的不确定性量化
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 基础ASR（用于特征提取和后处理）
        self.base_asr = ChineseASR(config)
        
        # Conformal ASR（用于不确定性量化）
        self.conformal_asr = ConformalASR(
            base_model=config.get('model_name', 'large-v3').split('/')[-1],
            coverage=config.get('conformal_coverage', 0.95),
            language=config.get('language', 'zh'),
            beam_size=config.get('beam_size', 5)
        )
        
        self.use_conformal = config.get('use_conformal', True)
        self.sample_rate = config.get('sample_rate', 16000)
        
    def forward(
        self, 
        audio_path: str,
        return_alternatives: bool = True
    ) -> ConformalASROutput:
        """
        增强的ASR处理流程
        
        Args:
            audio_path: 音频文件路径
            return_alternatives: 是否返回替代转录
            
        Returns:
            带Conformal预测的ASR输出
        """
        try:
            # 1. 加载和预处理音频
            audio = self.base_asr.preprocess_audio(audio_path)
            
            # 2. Conformal预测（如果启用）
            if self.use_conformal:
                pred_set = self.conformal_asr.predict(audio)
                
                # 使用top预测作为主文本
                main_text = pred_set.top_prediction
                prediction_set = pred_set.prediction_set
                set_size = pred_set.set_size
                conformal_conf = 1.0 / set_size
                coverage = pred_set.coverage_probability
            else:
                # 标准Whisper转录
                result = self.base_asr.model.transcribe(
                    audio,
                    language=self.base_asr.language,
                    task='transcribe',
                    word_timestamps=True,
                    verbose=False
                )
                main_text = result['text']
                prediction_set = [main_text]
                set_size = 1
                conformal_conf = 1.0
                coverage = 1.0
                
            # 3. 获取详细的segments（用于停顿检测）
            result = self.base_asr.model.transcribe(
                audio,
                language=self.base_asr.language,
                task='transcribe',
                word_timestamps=True,
                verbose=False
            )
            segments = result.get('segments', [])
            
            # 4. 停顿检测
            pause_info = self.base_asr.detect_pauses(audio, segments)
            
            # 5. 提取声学特征
            acoustic_features = self.base_asr.extract_acoustic_features(audio)
            
            # 6. 置信度分数
            confidence_scores = [seg.get('avg_logprob', 0.0) for seg in segments]
            
            # 7. 计算语速
            audio_duration = len(audio) / self.sample_rate
            speech_rate = self.base_asr.calculate_speech_rate(
                main_text, audio_duration, pause_info
            )
            
            enhanced_pause_info = {
                **pause_info,
                'speech_rate': speech_rate,
                'audio_duration': audio_duration
            }
            
            return ConformalASROutput(
                text=main_text,
                segments=segments,
                pause_info=enhanced_pause_info,
                acoustic_features=acoustic_features,
                confidence_scores=confidence_scores,
                prediction_set=prediction_set,
                set_size=set_size,
                conformal_confidence=conformal_conf,
                coverage_probability=coverage
            )
            
        except Exception as e:
            logger.error(f"Conformal ASR处理失败: {e}")
            # 返回空结果
            return ConformalASROutput(
                text="",
                segments=[],
                pause_info={},
                acoustic_features=torch.zeros(1, self.config['feature_dim']),
                confidence_scores=[],
                prediction_set=[""],
                set_size=1,
                conformal_confidence=0.0,
                coverage_probability=0.0
            )
    
    def calibrate(
        self,
        calibration_audio_paths: List[str],
        calibration_texts: List[str]
    ):
        """
        校准Conformal预测器
        
        Args:
            calibration_audio_paths: 校准音频路径列表
            calibration_texts: 对应的真实文本
        """
        logger.info(f"开始校准Conformal ASR...")
        
        # 加载音频
        calibration_audios = []
        for audio_path in calibration_audio_paths:
            audio = self.base_asr.preprocess_audio(audio_path)
            calibration_audios.append(audio)
        
        # 校准
        self.conformal_asr.calibrate(calibration_audios, calibration_texts)
        
        logger.info("Conformal ASR校准完成!")
        
    def compare_predictions(
        self,
        audio_path: str,
        ground_truth: str
    ) -> Dict:
        """
        比较有无Conformal的预测结果
        
        Args:
            audio_path: 音频文件
            ground_truth: 真实转录
            
        Returns:
            比较结果
        """
        # 无Conformal预测
        self.use_conformal = False
        result_without = self.forward(audio_path)
        
        # 有Conformal预测
        self.use_conformal = True
        result_with = self.forward(audio_path)
        
        # 计算准确性
        def compute_accuracy(pred: str, truth: str) -> float:
            # 简单的字符级准确率
            pred_chars = list(pred.replace(" ", ""))
            truth_chars = list(truth.replace(" ", ""))
            
            # 计算编辑距离
            m, n = len(pred_chars), len(truth_chars)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
                
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if pred_chars[i-1] == truth_chars[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
            
            edit_distance = dp[m][n]
            accuracy = 1 - (edit_distance / max(m, n))
            
            return accuracy
        
        acc_without = compute_accuracy(result_without.text, ground_truth)
        acc_with = compute_accuracy(result_with.text, ground_truth)
        
        # 检查真实文本是否在预测集中
        is_covered = ground_truth in result_with.prediction_set
        
        return {
            'ground_truth': ground_truth,
            'prediction_without_conformal': result_without.text,
            'prediction_with_conformal': result_with.text,
            'prediction_set': result_with.prediction_set,
            'set_size': result_with.set_size,
            'accuracy_without': acc_without,
            'accuracy_with': acc_with,
            'improvement': acc_with - acc_without,
            'is_covered': is_covered,
            'conformal_confidence': result_with.conformal_confidence
        }
    
    def batch_process(
        self,
        audio_paths: List[str]
    ) -> List[ConformalASROutput]:
        """批量处理音频文件"""
        
        results = []
        for audio_path in audio_paths:
            result = self.forward(audio_path)
            results.append(result)
            
        return results


def create_conformal_enhanced_asr(config: Dict) -> ConformalEnhancedASR:
    """创建增强ASR实例"""
    
    model = ConformalEnhancedASR(config)
    
    logger.info(f"创建Conformal增强ASR")
    logger.info(f"  基础模型: {config.get('model_name')}")
    logger.info(f"  Conformal覆盖率: {config.get('conformal_coverage', 0.95):.2%}")
    logger.info(f"  使用Conformal: {config.get('use_conformal', True)}")
    
    return model

