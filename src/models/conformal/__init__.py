"""
Conformal Inference 模块

为ASR预测提供严格的统计保证，生成预测集而非单一预测
增强系统对ASR错误的鲁棒性
"""

from .conformal_asr import ConformalASR, ConformalPredictionSet
from .calibrator import ConformalCalibrator

__all__ = [
    'ConformalASR',
    'ConformalPredictionSet',
    'ConformalCalibrator'
]

