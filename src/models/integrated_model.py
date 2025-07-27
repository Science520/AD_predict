import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging

from .asr.transformer_asr import TransformerASR
from .concepts.concept_extractor import ConceptBottleneckLayer
from .crf.crf_classifier import CRFClassifier, SimpleClassifier

logger = logging.getLogger(__name__)


class IntegratedAlzheimerModel(nn.Module):
    """集成的阿尔茨海默症检测模型
    
    整合多模态特征提取、概念瓶颈层和CRF分类器
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 第一阶段：多模态特征提取
        self.asr_model = TransformerASR(config['asr'])
        self.eeg_processor = EEGFeatureExtractor(config['eeg'])
        
        # 第二阶段：概念瓶颈层 (核心创新!)
        self.concept_layer = ConceptBottleneckLayer(config)
        
        # 第三阶段：分类器
        if config['crf'].get('use_crf', True):
            self.classifier = CRFClassifier(config['crf'])
        else:
            self.classifier = SimpleClassifier(config['crf'])
        
        # 损失权重
        self.loss_weights = config.get('loss_weights', {
            'diagnosis_loss': 1.0,
            'concept_loss': 0.5,
            'consistency_loss': 0.1
        })
        
        logger.info("初始化集成阿尔茨海默症检测模型")
    
    def forward(
        self, 
        batch: Dict[str, torch.Tensor],
        return_concepts: bool = True,
        return_explanations: bool = False
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            batch: 输入批次数据
                - audio_features: 音频特征 [B, T, D] 或音频路径
                - eeg_features: EEG特征 [B, T, C, D]
                - text_features: 文本特征 [B, D]
                - concepts: 概念标签 (训练时)
                - diagnosis: 诊断标签 (训练时)
            return_concepts: 是否返回概念预测
            return_explanations: 是否生成解释
            
        Returns:
            outputs: 模型输出字典
        """
        # 阶段1: 多模态特征提取
        features = self._extract_features(batch)
        
        # 阶段2: 概念预测 (可解释性的关键!)
        concepts = self.concept_layer(features)
        
        # 阶段3: 基于概念的诊断
        diagnosis_targets = batch.get('diagnosis', None)
        classifier_outputs = self.classifier(concepts, diagnosis_targets)
        
        # 整合输出
        outputs = {
            'diagnosis_logits': classifier_outputs['logits'],
            'diagnosis_probs': classifier_outputs['probabilities'],
            'diagnosis_predictions': classifier_outputs['predictions']
        }
        
        if return_concepts:
            outputs['concepts'] = concepts
        
        # 计算损失 (训练时)
        if self.training:
            total_loss, loss_components = self._compute_losses(
                diagnosis_outputs=classifier_outputs,
                concept_predictions=concepts,
                targets=batch
            )
            outputs['total_loss'] = total_loss
            outputs['loss_components'] = loss_components
        
        # 生成解释 (如果需要)
        if return_explanations:
            explanations = self._generate_explanations(concepts, classifier_outputs)
            outputs['explanations'] = explanations
        
        return outputs
    
    def _extract_features(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """提取多模态特征"""
        features = {}
        
        # ASR处理：音频 -> 文本 + 声学特征
        if 'audio_features' in batch:
            audio_input = batch['audio_features']
            asr_outputs = self.asr_model(audio_input)
            
            features['speech'] = asr_outputs['acoustic_features']
            features['text'] = asr_outputs['text_features']
        else:
            # 如果没有音频，使用预提取的特征
            if 'text_features' in batch:
                features['text'] = batch['text_features']
            else:
                # 创建零特征
                batch_size = batch['eeg_features'].shape[0] if 'eeg_features' in batch else 1
                device = next(self.parameters()).device
                features['text'] = torch.zeros(batch_size, 768, device=device)
            
            if 'speech_features' in batch:
                features['speech'] = batch['speech_features']
            else:
                batch_size = features['text'].shape[0]
                device = features['text'].device
                features['speech'] = torch.zeros(batch_size, 1000, 768, device=device)
        
        # EEG特征处理
        if 'eeg_features' in batch:
            features['eeg'] = batch['eeg_features']
        else:
            # 创建零EEG特征
            batch_size = features['text'].shape[0]
            device = features['text'].device
            features['eeg'] = torch.zeros(batch_size, 100, 19, 15, device=device)
        
        return features
    
    def _compute_losses(
        self,
        diagnosis_outputs: Dict[str, torch.Tensor],
        concept_predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算总损失"""
        loss_components = {}
        
        # 1. 诊断损失
        if 'diagnosis' in targets:
            if 'crf_loss' in diagnosis_outputs:
                # 使用CRF损失
                diagnosis_loss = diagnosis_outputs['crf_loss']
            else:
                # 使用交叉熵损失
                diagnosis_loss = F.cross_entropy(
                    diagnosis_outputs['logits'], 
                    targets['diagnosis']
                )
            loss_components['diagnosis_loss'] = diagnosis_loss
        else:
            loss_components['diagnosis_loss'] = torch.tensor(0.0, device=next(self.parameters()).device)
        
        # 2. 概念损失
        if 'concepts' in targets:
            concept_loss, individual_concept_losses = self.concept_layer.compute_concept_loss(
                concept_predictions, targets['concepts']
            )
            loss_components['concept_loss'] = concept_loss
            loss_components.update(individual_concept_losses)
        else:
            loss_components['concept_loss'] = torch.tensor(0.0, device=next(self.parameters()).device)
        
        # 3. 一致性损失 (概念和诊断的一致性)
        consistency_loss = self._compute_consistency_loss(
            concept_predictions, diagnosis_outputs['probabilities']
        )
        loss_components['consistency_loss'] = consistency_loss
        
        # 计算总损失
        total_loss = (
            self.loss_weights['diagnosis_loss'] * loss_components['diagnosis_loss'] +
            self.loss_weights['concept_loss'] * loss_components['concept_loss'] +
            self.loss_weights['consistency_loss'] * loss_components['consistency_loss']
        )
        
        return total_loss, loss_components
    
    def _compute_consistency_loss(
        self,
        concepts: Dict[str, torch.Tensor],
        diagnosis_probs: torch.Tensor
    ) -> torch.Tensor:
        """计算概念和诊断之间的一致性损失"""
        try:
            # 基于医学知识的一致性约束
            consistency_loss = 0.0
            
            # 例子：语速慢 + 停顿多 + 词汇丰富度低 应该倾向于AD诊断
            if 'speech_rate' in concepts and 'pause_ratio' in concepts and 'lexical_richness' in concepts:
                speech_rate = concepts['speech_rate'].squeeze(-1)  # [B]
                pause_ratio = concepts['pause_ratio'].squeeze(-1)  # [B]
                lexical_richness = concepts['lexical_richness'].squeeze(-1)  # [B]
                
                # 计算认知衰退指标
                cognitive_decline = (
                    (1.0 - torch.sigmoid(speech_rate - 2.0)) +  # 语速慢
                    torch.sigmoid(pause_ratio - 0.3) +         # 停顿多
                    (1.0 - lexical_richness)                   # 词汇贫乏
                ) / 3.0
                
                # AD类别的概率应该与认知衰退指标相关
                ad_prob = diagnosis_probs[:, 2]  # 假设AD是第3类 (index=2)
                
                # 相关性损失
                correlation_loss = F.mse_loss(ad_prob, cognitive_decline)
                consistency_loss += correlation_loss
            
            # EEG一致性：Alpha波功率低 + Theta/Beta比值高 应该倾向于AD
            if 'alpha_power' in concepts and 'theta_beta_ratio' in concepts:
                alpha_power = concepts['alpha_power'].squeeze(-1)
                theta_beta_ratio = concepts['theta_beta_ratio'].squeeze(-1)
                
                # 归一化到[0,1]
                alpha_normalized = alpha_power / 100.0
                theta_beta_normalized = torch.clamp(theta_beta_ratio / 10.0, 0, 1)
                
                # EEG异常指标
                eeg_abnormality = (
                    (1.0 - alpha_normalized) +  # Alpha功率低
                    theta_beta_normalized        # Theta/Beta比值高
                ) / 2.0
                
                ad_prob = diagnosis_probs[:, 2]
                eeg_consistency_loss = F.mse_loss(ad_prob, eeg_abnormality)
                consistency_loss += eeg_consistency_loss
            
            return consistency_loss
            
        except Exception as e:
            logger.warning(f"一致性损失计算失败: {e}")
            return torch.tensor(0.0, device=diagnosis_probs.device)
    
    def _generate_explanations(
        self,
        concepts: Dict[str, torch.Tensor],
        diagnosis_outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, any]:
        """生成可解释性解释"""
        explanations = {}
        
        # 概念解释
        concept_explanations = self.concept_layer.get_concept_explanations(concepts)
        explanations['concept_explanations'] = concept_explanations
        
        # 概念重要性
        concept_importance = self.concept_layer.get_concept_importance(concepts)
        explanations['concept_importance'] = concept_importance
        
        # 诊断解释 (如果支持)
        if hasattr(self.classifier, 'explain_prediction'):
            batch_size = diagnosis_outputs['predictions'].shape[0]
            diagnosis_explanations = []
            
            for i in range(batch_size):
                # 提取单个样本的概念
                sample_concepts = {
                    name: value[i:i+1] for name, value in concepts.items()
                }
                prediction = diagnosis_outputs['predictions'][i].item()
                
                explanation = self.classifier.explain_prediction(sample_concepts, prediction)
                diagnosis_explanations.append(explanation)
            
            explanations['diagnosis_explanations'] = diagnosis_explanations
        
        return explanations
    
    def predict(
        self,
        audio_input: Optional[torch.Tensor] = None,
        eeg_input: Optional[torch.Tensor] = None,
        text_input: Optional[torch.Tensor] = None,
        return_explanations: bool = True
    ) -> Dict[str, any]:
        """单样本预测接口
        
        Args:
            audio_input: 音频输入
            eeg_input: EEG输入
            text_input: 文本输入
            return_explanations: 是否返回解释
            
        Returns:
            预测结果和解释
        """
        self.eval()
        
        # 准备批次数据
        batch = {}
        if audio_input is not None:
            if audio_input.dim() == 1:
                audio_input = audio_input.unsqueeze(0)
            batch['audio_features'] = audio_input
        
        if eeg_input is not None:
            if eeg_input.dim() == 3:
                eeg_input = eeg_input.unsqueeze(0)
            batch['eeg_features'] = eeg_input
        
        if text_input is not None:
            if text_input.dim() == 1:
                text_input = text_input.unsqueeze(0)
            batch['text_features'] = text_input
        
        # 前向传播
        with torch.no_grad():
            outputs = self.forward(
                batch, 
                return_concepts=True, 
                return_explanations=return_explanations
            )
        
        # 整理结果
        prediction = outputs['diagnosis_predictions'][0].item()
        probabilities = outputs['diagnosis_probs'][0]
        
        result = {
            'predicted_class': self.config['crf']['class_names'][prediction],
            'prediction_index': prediction,
            'class_probabilities': {
                name: prob.item() 
                for name, prob in zip(self.config['crf']['class_names'], probabilities)
            },
            'confidence': probabilities[prediction].item()
        }
        
        # 添加概念信息
        if 'concepts' in outputs:
            result['concepts'] = {
                name: value[0].item() 
                for name, value in outputs['concepts'].items()
            }
        
        # 添加解释
        if return_explanations and 'explanations' in outputs:
            result['explanations'] = outputs['explanations']
        
        return result


class EEGFeatureExtractor(nn.Module):
    """EEG特征提取器 (简化版本)"""
    
    def __init__(self, eeg_config: Dict):
        super().__init__()
        self.config = eeg_config
        
        # 这里可以集成更复杂的EEG处理模型
        # 现在使用简单的占位符
        self.feature_dim = eeg_config.get('feature_dim', 512)
        
    def forward(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_data: EEG数据 [B, T, C, raw_features]
            
        Returns:
            eeg_features: 处理后的EEG特征 [B, T, C, D]
        """
        # 简单的线性变换 (实际应用中会更复杂)
        if eeg_data.shape[-1] != self.feature_dim:
            batch_size, seq_len, n_channels, raw_dim = eeg_data.shape
            
            # 简单的特征变换
            linear_transform = nn.Linear(raw_dim, self.feature_dim).to(eeg_data.device)
            eeg_reshaped = eeg_data.view(-1, raw_dim)
            transformed = linear_transform(eeg_reshaped)
            eeg_features = transformed.view(batch_size, seq_len, n_channels, self.feature_dim)
        else:
            eeg_features = eeg_data
        
        return eeg_features 