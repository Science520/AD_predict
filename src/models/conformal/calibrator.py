#!/usr/bin/env python3
"""
Conformal校准器

提供多种校准策略和验证方法
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ConformalCalibrator:
    """
    Conformal预测校准器
    
    提供不同的校准策略和有效性验证
    """
    
    def __init__(self, coverage: float = 0.95):
        self.coverage = coverage
        self.alpha = 1 - coverage
        
    def calibrate_standard(
        self,
        scores: np.ndarray
    ) -> float:
        """
        标准校准方法
        
        使用 (n+1)(1-alpha) / n 分位数
        """
        n = len(scores)
        q = np.ceil((n + 1) * self.coverage) / n
        threshold = np.quantile(scores, q)
        
        return threshold
    
    def calibrate_adaptive(
        self,
        scores: np.ndarray,
        difficulty_scores: Optional[np.ndarray] = None
    ) -> float:
        """
        自适应校准
        
        根据样本难度调整阈值
        """
        if difficulty_scores is None:
            return self.calibrate_standard(scores)
        
        # 根据难度加权
        weights = 1.0 / (difficulty_scores + 1e-6)
        weights = weights / weights.sum()
        
        # 加权分位数
        sorted_indices = np.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        cumsum_weights = np.cumsum(sorted_weights)
        idx = np.searchsorted(cumsum_weights, self.coverage)
        
        threshold = sorted_scores[min(idx, len(sorted_scores) - 1)]
        
        return threshold
    
    def validate_coverage(
        self,
        test_scores: np.ndarray,
        threshold: float,
        test_labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        验证校准后的覆盖率
        
        Args:
            test_scores: 测试集的非一致性分数
            threshold: 校准阈值
            test_labels: 测试标签（可选）
            
        Returns:
            验证统计信息
        """
        # 计算实际覆盖率
        covered = (test_scores <= threshold).sum()
        actual_coverage = covered / len(test_scores)
        
        # 平均预测集大小（这里简化为二分类）
        avg_set_size = actual_coverage
        
        stats = {
            'actual_coverage': actual_coverage,
            'target_coverage': self.coverage,
            'coverage_gap': actual_coverage - self.coverage,
            'avg_set_size': avg_set_size,
            'n_samples': len(test_scores)
        }
        
        logger.info(f"验证结果:")
        logger.info(f"  目标覆盖率: {self.coverage:.2%}")
        logger.info(f"  实际覆盖率: {actual_coverage:.2%}")
        logger.info(f"  覆盖率差距: {stats['coverage_gap']:.2%}")
        
        return stats
    
    def compute_efficiency(
        self,
        prediction_set_sizes: List[int]
    ) -> Dict[str, float]:
        """
        计算Conformal预测的效率
        
        效率越高，预测集越小，说明模型越确定
        """
        sizes = np.array(prediction_set_sizes)
        
        return {
            'avg_set_size': np.mean(sizes),
            'median_set_size': np.median(sizes),
            'max_set_size': np.max(sizes),
            'min_set_size': np.min(sizes),
            'std_set_size': np.std(sizes),
            'singleton_rate': (sizes == 1).mean()  # 单元素集合的比例
        }

