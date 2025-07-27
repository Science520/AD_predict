"""
评估指标计算工具
Metrics calculation utilities for Alzheimer detection system
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from typing import Dict, List, Any
import torch


def compute_classification_metrics(
    y_true: List[int], 
    y_pred: List[int], 
    class_names: List[str] = None
) -> Dict[str, float]:
    """计算分类指标"""
    
    if class_names is None:
        class_names = ["Healthy", "MCI", "AD"]
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
    }
    
    # 每个类别的指标
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
    
    return metrics


def compute_concept_metrics(
    concept_true: Dict[str, List[float]], 
    concept_pred: Dict[str, List[float]]
) -> Dict[str, Dict[str, float]]:
    """计算概念预测指标"""
    
    concept_metrics = {}
    
    for concept_name in concept_true.keys():
        if concept_name in concept_pred:
            true_values = np.array(concept_true[concept_name])
            pred_values = np.array(concept_pred[concept_name])
            
            # 确保形状匹配
            min_len = min(len(true_values), len(pred_values))
            true_values = true_values[:min_len]
            pred_values = pred_values[:min_len]
            
            concept_metrics[concept_name] = {
                'mae': mean_absolute_error(true_values, pred_values),
                'mse': mean_squared_error(true_values, pred_values),
                'rmse': np.sqrt(mean_squared_error(true_values, pred_values)),
                'r2': r2_score(true_values, pred_values) if len(set(true_values)) > 1 else 0.0,
                'correlation': np.corrcoef(true_values, pred_values)[0, 1] if len(true_values) > 1 else 0.0
            }
    
    return concept_metrics


def compute_consistency_metrics(
    concept_pred: Dict[str, torch.Tensor],
    diagnosis_pred: torch.Tensor,
    consistency_rules: Dict[str, Any] = None
) -> Dict[str, float]:
    """计算概念和诊断一致性指标"""
    
    if consistency_rules is None:
        # 默认一致性规则
        consistency_rules = {
            'alpha_power_ad_negative': True,  # AD患者alpha功率通常较低
            'theta_beta_ratio_ad_positive': True,  # AD患者theta/beta比值通常较高
            'speech_rate_ad_negative': True,  # AD患者语速通常较慢
        }
    
    consistency_metrics = {}
    
    # 转换为numpy数组
    concept_values = {}
    for name, tensor in concept_pred.items():
        concept_values[name] = tensor.detach().cpu().numpy()
    
    diagnosis_values = diagnosis_pred.detach().cpu().numpy()
    
    # 计算各种一致性指标
    # 这里可以实现基于医学知识的一致性检查
    
    # 示例：检查AD诊断与概念的一致性
    if 'alpha_power' in concept_values and len(diagnosis_values.shape) > 1:
        ad_prob = diagnosis_values[:, -1] if diagnosis_values.shape[1] >= 3 else np.zeros_like(diagnosis_values[:, 0])
        alpha_power = concept_values['alpha_power']
        
        # AD概率高时，alpha功率应该低
        consistency_score = np.corrcoef(-alpha_power, ad_prob)[0, 1] if len(alpha_power) > 1 else 0.0
        consistency_metrics['alpha_ad_consistency'] = consistency_score
    
    return consistency_metrics 