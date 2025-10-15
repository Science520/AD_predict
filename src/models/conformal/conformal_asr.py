#!/usr/bin/env python3
"""
Conformal Inference for ASR

使用Conformal Prediction为ASR提供不确定性量化
生成包含多个候选转录的预测集，并提供统计保证
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import logging
from pathlib import Path
import whisper

logger = logging.getLogger(__name__)


@dataclass
class ConformalPredictionSet:
    """Conformal预测集"""
    prediction_set: List[str]  # 候选转录文本集合
    calibrated_score: float    # 校准后的分数阈值
    coverage_probability: float  # 覆盖概率 (e.g., 0.95)
    top_prediction: str  # 最优预测
    prediction_scores: List[float]  # 每个候选的得分
    set_size: int  # 预测集大小
    
    
class ConformalASR:
    """
    Conformal ASR - 带不确定性量化的ASR系统
    
    核心思想：
    1. 使用校准集计算非一致性分数
    2. 根据期望覆盖率确定阈值
    3. 生成包含多个候选的预测集，保证真实转录以高概率在集合内
    """
    
    def __init__(
        self,
        base_model: str = "large-v3",
        coverage: float = 0.95,
        language: str = "zh",
        beam_size: int = 5
    ):
        """
        Args:
            base_model: Whisper模型名称
            coverage: 期望覆盖概率 (1-alpha)
            language: 语言
            beam_size: beam search的宽度
        """
        self.base_model = whisper.load_model(base_model)
        self.coverage = coverage
        self.language = language
        self.beam_size = beam_size
        
        # 校准阈值（通过校准集确定）
        self.calibrated_threshold = None
        self.is_calibrated = False
        
    def compute_nonconformity_score(
        self,
        audio: np.ndarray,
        true_text: Optional[str] = None
    ) -> Tuple[float, List[Tuple[str, float]]]:
        """
        计算非一致性分数
        
        非一致性分数衡量预测的"奇异程度"
        分数越高，预测越不可靠
        
        Args:
            audio: 音频数组
            true_text: 真实文本（校准时使用）
            
        Returns:
            nonconformity_score: 非一致性分数
            candidates: 候选列表 [(text, score), ...]
        """
        # Beam search获取多个候选
        result = self.base_model.transcribe(
            audio,
            language=self.language,
            task='transcribe',
            beam_size=self.beam_size,
            best_of=self.beam_size,
            temperature=0.0,
            verbose=False
        )
        
        # 获取候选及其对数概率
        # Whisper默认返回最佳结果，我们需要访问内部来获取beam search的所有候选
        # 这里我们使用一个简化版本：通过温度采样获得多样性
        candidates = []
        
        # 主预测
        main_text = result['text']
        main_logprob = np.mean([seg.get('avg_logprob', 0.0) 
                                for seg in result.get('segments', [])])
        candidates.append((main_text, main_logprob))
        
        # 生成额外候选（使用不同温度）
        for temp in [0.2, 0.4, 0.6, 0.8]:
            try:
                alt_result = self.base_model.transcribe(
                    audio,
                    language=self.language,
                    task='transcribe',
                    temperature=temp,
                    verbose=False
                )
                alt_text = alt_result['text']
                alt_logprob = np.mean([seg.get('avg_logprob', 0.0) 
                                      for seg in alt_result.get('segments', [])])
                
                # 避免重复
                if alt_text not in [c[0] for c in candidates]:
                    candidates.append((alt_text, alt_logprob))
            except:
                continue
        
        # 按分数排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 计算非一致性分数
        if true_text is not None:
            # 校准模式：分数 = 1 - P(true_text)
            # 找到真实文本的排名
            texts = [c[0] for c in candidates]
            
            if true_text in texts:
                true_idx = texts.index(true_text)
                true_score = candidates[true_idx][1]
            else:
                # 真实文本不在候选中，给予最低分
                true_score = candidates[-1][1] - 1.0
            
            # 非一致性分数：越小越好
            nonconformity_score = -true_score
        else:
            # 预测模式：使用top-1的负对数概率
            nonconformity_score = -candidates[0][1]
        
        return nonconformity_score, candidates
    
    def calibrate(
        self,
        calibration_audios: List[np.ndarray],
        calibration_texts: List[str]
    ):
        """
        使用校准集确定阈值
        
        Args:
            calibration_audios: 校准音频列表
            calibration_texts: 对应的真实文本
        """
        logger.info(f"开始校准Conformal ASR，校准集大小: {len(calibration_audios)}")
        
        # 计算所有校准样本的非一致性分数
        scores = []
        for audio, true_text in zip(calibration_audios, calibration_texts):
            score, _ = self.compute_nonconformity_score(audio, true_text)
            scores.append(score)
        
        scores = np.array(scores)
        
        # 确定分位数阈值
        # 对于coverage = 0.95, 我们取 ceil((n+1) * 0.95) / n 分位数
        n = len(scores)
        q = np.ceil((n + 1) * self.coverage) / n
        self.calibrated_threshold = np.quantile(scores, q)
        
        self.is_calibrated = True
        
        logger.info(f"校准完成! 阈值: {self.calibrated_threshold:.4f}")
        logger.info(f"期望覆盖率: {self.coverage:.2%}")
        
    def predict(
        self,
        audio: np.ndarray
    ) -> ConformalPredictionSet:
        """
        生成Conformal预测集
        
        Args:
            audio: 输入音频
            
        Returns:
            包含多个候选的预测集，保证真实转录以coverage概率在集合中
        """
        if not self.is_calibrated:
            logger.warning("模型未校准，将返回所有候选")
            threshold = float('inf')
        else:
            threshold = self.calibrated_threshold
        
        # 计算非一致性分数和候选
        score, candidates = self.compute_nonconformity_score(audio, None)
        
        # 构建预测集：包含所有非一致性分数 <= threshold 的候选
        prediction_set = []
        prediction_scores = []
        
        for text, logprob in candidates:
            candidate_score = -logprob  # 非一致性分数
            
            if candidate_score <= threshold:
                prediction_set.append(text)
                prediction_scores.append(float(logprob))
        
        # 如果预测集为空（理论上不应该发生），至少包含top-1
        if len(prediction_set) == 0:
            prediction_set = [candidates[0][0]]
            prediction_scores = [float(candidates[0][1])]
        
        return ConformalPredictionSet(
            prediction_set=prediction_set,
            calibrated_score=float(threshold),
            coverage_probability=self.coverage,
            top_prediction=candidates[0][0],
            prediction_scores=prediction_scores,
            set_size=len(prediction_set)
        )
    
    def predict_with_confidence(
        self,
        audio: np.ndarray
    ) -> Tuple[str, float, ConformalPredictionSet]:
        """
        预测并返回置信度
        
        Args:
            audio: 输入音频
            
        Returns:
            (预测文本, 置信度, 预测集)
            置信度基于预测集大小：集合越小，置信度越高
        """
        pred_set = self.predict(audio)
        
        # 置信度：1 / set_size
        # 预测集越小，说明模型越确定
        confidence = 1.0 / pred_set.set_size
        
        return pred_set.top_prediction, confidence, pred_set
    
    def batch_predict(
        self,
        audios: List[np.ndarray]
    ) -> List[ConformalPredictionSet]:
        """批量预测"""
        
        results = []
        for audio in audios:
            pred_set = self.predict(audio)
            results.append(pred_set)
            
        return results
    
    def save_calibration(self, save_path: str):
        """保存校准参数"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(save_path, {
            'calibrated_threshold': self.calibrated_threshold,
            'coverage': self.coverage,
            'is_calibrated': self.is_calibrated
        })
        
        logger.info(f"校准参数已保存到: {save_path}")
        
    def load_calibration(self, load_path: str):
        """加载校准参数"""
        data = np.load(load_path, allow_pickle=True).item()
        
        self.calibrated_threshold = data['calibrated_threshold']
        self.coverage = data['coverage']
        self.is_calibrated = data['is_calibrated']
        
        logger.info(f"校准参数已从 {load_path} 加载")


def create_conformal_asr(
    model_name: str = "large-v3",
    coverage: float = 0.95,
    language: str = "zh"
) -> ConformalASR:
    """创建Conformal ASR实例"""
    
    model = ConformalASR(
        base_model=model_name,
        coverage=coverage,
        language=language
    )
    
    logger.info(f"创建Conformal ASR: {model_name}")
    logger.info(f"目标覆盖率: {coverage:.2%}")
    
    return model

