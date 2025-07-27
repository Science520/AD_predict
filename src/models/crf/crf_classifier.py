import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class CRFClassifier(nn.Module):
    """基于概念的CRF分类器
    
    使用概念瓶颈层的输出进行最终的阿尔茨海默症诊断
    """
    
    def __init__(self, crf_config: Dict):
        super().__init__()
        self.config = crf_config
        self.num_classes = crf_config['num_classes']
        self.class_names = crf_config['class_names']
        
        # 概念特征维度 (根据概念数量确定)
        self.concept_dim = self._calculate_concept_dim()
        
        # 概念特征处理层
        self.concept_processor = ConceptProcessor(
            concept_dim=self.concept_dim,
            hidden_dim=256
        )
        
        # 特征增强层
        self.feature_enhancer = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
        )
        
        # CRF层 (用于序列标注，这里简化为单步预测)
        self.crf = CRF(self.num_classes, batch_first=True)
        
        # 发射分数计算
        self.emission_layer = nn.Linear(256, self.num_classes)
        
        # 概念重要性权重
        self.concept_weights = nn.Parameter(torch.ones(self.concept_dim))
        
        logger.info(f"初始化CRF分类器: {self.num_classes} 类, 概念维度: {self.concept_dim}")
    
    def _calculate_concept_dim(self) -> int:
        """计算概念特征维度"""
        # 这里假设有固定数量的概念
        # 实际使用时需要根据配置动态计算
        speech_concepts = 4  # speech_rate, pause_ratio, lexical_richness, syntactic_complexity
        eeg_concepts = 3     # alpha_power, theta_beta_ratio, gamma_connectivity
        
        return speech_concepts + eeg_concepts
    
    def forward(
        self, 
        concepts: Dict[str, torch.Tensor],
        targets: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            concepts: 概念特征字典 {concept_name: [B, 1]}
            targets: 目标标签 [B] (训练时使用)
            return_features: 是否返回中间特征
            
        Returns:
            Dict包含:
            - logits: 分类logits [B, num_classes]
            - predictions: 预测类别 [B]
            - probabilities: 类别概率 [B, num_classes]
            - concept_importance: 概念重要性 [concept_dim]
        """
        batch_size = list(concepts.values())[0].shape[0]
        
        # 处理概念特征
        concept_features = self._process_concepts(concepts)  # [B, concept_dim]
        
        # 应用概念权重
        weighted_concepts = concept_features * self.concept_weights.unsqueeze(0)  # [B, concept_dim]
        
        # 概念特征处理
        processed_features = self.concept_processor(weighted_concepts)  # [B, 256]
        
        # 特征增强
        enhanced_features = self.feature_enhancer(processed_features)  # [B, 256]
        
        # 计算发射分数
        emissions = self.emission_layer(enhanced_features)  # [B, num_classes]
        
        # 由于我们是单步预测，将其扩展为序列
        emissions_seq = emissions.unsqueeze(1)  # [B, 1, num_classes]
        
        if self.training and targets is not None:
            # 训练时计算CRF损失
            targets_seq = targets.unsqueeze(1)  # [B, 1]
            crf_loss = -self.crf(emissions_seq, targets_seq)
            
            # 预测
            predictions_seq = self.crf.decode(emissions_seq)
            predictions = torch.tensor([pred[0] for pred in predictions_seq], 
                                     device=emissions.device)
        else:
            # 推理时只做预测
            predictions_seq = self.crf.decode(emissions_seq)
            predictions = torch.tensor([pred[0] for pred in predictions_seq], 
                                     device=emissions.device)
            crf_loss = None
        
        # 计算概率
        probabilities = F.softmax(emissions, dim=1)  # [B, num_classes]
        
        # 概念重要性 (归一化权重)
        concept_importance = F.softmax(self.concept_weights, dim=0)
        
        outputs = {
            'logits': emissions,
            'predictions': predictions,
            'probabilities': probabilities,
            'concept_importance': concept_importance
        }
        
        if crf_loss is not None:
            outputs['crf_loss'] = crf_loss
        
        if return_features:
            outputs['concept_features'] = concept_features
            outputs['processed_features'] = processed_features
            outputs['enhanced_features'] = enhanced_features
        
        return outputs
    
    def _process_concepts(self, concepts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """处理概念特征"""
        # 预定义的概念顺序
        concept_order = [
            'speech_rate', 'pause_ratio', 'lexical_richness', 'syntactic_complexity',
            'alpha_power', 'theta_beta_ratio', 'gamma_connectivity'
        ]
        
        concept_values = []
        for concept_name in concept_order:
            if concept_name in concepts:
                value = concepts[concept_name]
                if value.dim() > 1:
                    value = value.squeeze(-1)  # [B, 1] -> [B]
                concept_values.append(value)
            else:
                # 如果概念不存在，用零填充
                batch_size = list(concepts.values())[0].shape[0]
                device = list(concepts.values())[0].device
                concept_values.append(torch.zeros(batch_size, device=device))
        
        # 拼接所有概念
        concept_features = torch.stack(concept_values, dim=1)  # [B, concept_dim]
        
        return concept_features
    
    def get_concept_contributions(
        self, 
        concepts: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """计算每个概念对最终预测的贡献度"""
        concept_names = [
            'speech_rate', 'pause_ratio', 'lexical_richness', 'syntactic_complexity',
            'alpha_power', 'theta_beta_ratio', 'gamma_connectivity'
        ]
        
        # 获取概念重要性权重
        importance_weights = F.softmax(self.concept_weights, dim=0)
        
        contributions = {}
        for i, concept_name in enumerate(concept_names):
            if i < len(importance_weights):
                contributions[concept_name] = importance_weights[i].item()
            else:
                contributions[concept_name] = 0.0
        
        return contributions
    
    def explain_prediction(
        self, 
        concepts: Dict[str, torch.Tensor],
        prediction: int
    ) -> Dict[str, any]:
        """解释单个预测结果"""
        # 前向传播获取详细信息
        outputs = self.forward(concepts, return_features=True)
        
        # 概念贡献度
        concept_contributions = self.get_concept_contributions(concepts)
        
        # 预测置信度
        probabilities = outputs['probabilities'][0]  # 假设batch_size=1
        confidence = probabilities[prediction].item()
        
        # 关键概念识别 (贡献度最高的概念)
        top_concepts = sorted(concept_contributions.items(), 
                            key=lambda x: x[1], reverse=True)[:3]
        
        explanation = {
            'predicted_class': self.class_names[prediction],
            'confidence': confidence,
            'class_probabilities': {
                self.class_names[i]: prob.item() 
                for i, prob in enumerate(probabilities)
            },
            'concept_contributions': concept_contributions,
            'top_contributing_concepts': top_concepts,
            'concept_values': {
                name: concepts[name].item() if name in concepts else 0.0
                for name in concept_contributions.keys()
            }
        }
        
        return explanation


class ConceptProcessor(nn.Module):
    """概念特征处理器"""
    
    def __init__(self, concept_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.concept_dim = concept_dim
        self.hidden_dim = hidden_dim
        
        # 概念特征变换
        self.concept_transform = nn.Sequential(
            nn.Linear(concept_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 概念交互建模
        self.concept_interaction = nn.MultiheadAttention(
            embed_dim=concept_dim,
            num_heads=1,
            batch_first=True
        )
        
    def forward(self, concept_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            concept_features: 概念特征 [B, concept_dim]
            
        Returns:
            processed_features: 处理后的特征 [B, hidden_dim]
        """
        batch_size, concept_dim = concept_features.shape
        
        # 概念交互建模
        # 扩展维度以使用注意力机制
        concept_expanded = concept_features.unsqueeze(1)  # [B, 1, concept_dim]
        
        interacted_concepts, attention_weights = self.concept_interaction(
            concept_expanded, concept_expanded, concept_expanded
        )  # [B, 1, concept_dim]
        
        interacted_concepts = interacted_concepts.squeeze(1)  # [B, concept_dim]
        
        # 特征变换
        processed_features = self.concept_transform(interacted_concepts)  # [B, hidden_dim]
        
        return processed_features


class SimpleClassifier(nn.Module):
    """简化的分类器 (不使用CRF)"""
    
    def __init__(self, crf_config: Dict):
        super().__init__()
        self.config = crf_config
        self.num_classes = crf_config['num_classes']
        self.class_names = crf_config['class_names']
        
        # 概念特征维度
        self.concept_dim = 7  # 预定义概念数量
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.concept_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes)
        )
        
        logger.info(f"初始化简单分类器: {self.num_classes} 类")
    
    def forward(
        self, 
        concepts: Dict[str, torch.Tensor],
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 处理概念特征
        concept_features = self._process_concepts(concepts)
        
        # 分类
        logits = self.classifier(concept_features)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        
        outputs = {
            'logits': logits,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            outputs['loss'] = loss
        
        return outputs
    
    def _process_concepts(self, concepts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """处理概念特征"""
        concept_order = [
            'speech_rate', 'pause_ratio', 'lexical_richness', 'syntactic_complexity',
            'alpha_power', 'theta_beta_ratio', 'gamma_connectivity'
        ]
        
        concept_values = []
        for concept_name in concept_order:
            if concept_name in concepts:
                value = concepts[concept_name]
                if value.dim() > 1:
                    value = value.squeeze(-1)
                concept_values.append(value)
            else:
                batch_size = list(concepts.values())[0].shape[0]
                device = list(concepts.values())[0].device
                concept_values.append(torch.zeros(batch_size, device=device))
        
        concept_features = torch.stack(concept_values, dim=1)
        
        return concept_features 