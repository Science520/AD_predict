import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

from .concept_models import (
    SpeechRatePredictor,
    PauseRatioPredictor,
    LexicalRichnessPredictor,
    SyntacticComplexityPredictor,
    AlphaPowerPredictor,
    ThetaBetaRatioPredictor
)

logger = logging.getLogger(__name__)


class ConceptBottleneckLayer(nn.Module):
    """概念瓶颈层 - 系统的核心创新
    
    将底层多模态特征转换为可解释的医学概念，
    提供透明的诊断过程和可解释性
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.concept_configs = config['concepts']
        
        # 1. 定义通用概念和语言特定概念
        self.universal_concepts = {
            'acoustic': ['jitter', 'shimmer', 'pitch_variation'],
            'prosodic': ['pause_patterns', 'speech_rhythm'],
            'eeg': ['alpha_power', 'theta_beta_ratio']
        }
        
        self.language_specific_concepts = {
            'sino_tibetan': {
                'acoustic': ['tone_patterns'],
                'linguistic': ['classifier_usage', 'tone_errors'],
                'norms': {
                    'speech_rate': {'mean': 4.5, 'std': 0.8},  # 基于音节
                    'pause_ratio': {'mean': 0.18, 'std': 0.05},
                    'lexical_richness': {'mean': 0.45, 'std': 0.12},
                    'syntactic_complexity': {'mean': 3.8, 'std': 1.2}
                }
            },
            'indo_european': {
                'acoustic': ['formant_transitions'],
                'linguistic': ['article_usage', 'verb_agreement'],
                'norms': {
                    'speech_rate': {'mean': 2.8, 'std': 0.5},  # 基于词
                    'pause_ratio': {'mean': 0.22, 'std': 0.07},
                    'lexical_richness': {'mean': 0.48, 'std': 0.13},
                    'syntactic_complexity': {'mean': 4.2, 'std': 1.4}
                }
            }
        }
        
        # 2. 获取所有概念名称
        self.speech_concept_names = [c['name'] for c in self.concept_configs['speech_concepts']]
        self.eeg_concept_names = [c['name'] for c in self.concept_configs['eeg_concepts']]
        self.all_concept_names = self.speech_concept_names + self.eeg_concept_names
        
        # 3. 构建两阶段模型
        # 阶段一：特征到概念的映射
        self.concept_models = self._build_concept_models()
        
        # 阶段二：概念到诊断的映射
        self.diagnostic_model = self._build_diagnostic_model()
        
        # 4. 损失函数权重
        self.concept_loss_weight = config.get('concept_loss_weight', 1.0)  # 阶段一损失权重
        self.diagnostic_loss_weight = config.get('diagnostic_loss_weight', 1.0)  # 阶段二损失权重
        self.concept_regularization = config.get('concept_regularization', 0.01)  # 概念正则化权重
        
        logger.info(f"初始化概念瓶颈层: {len(self.all_concept_names)} 个概念")
        
    def _build_concept_models(self) -> nn.ModuleDict:
        """为每个概念构建预测模型"""
        models = nn.ModuleDict()
        
        # 语音概念模型
        for concept_config in self.concept_configs['speech_concepts']:
            concept_name = concept_config['name']
            
            if concept_name == 'speech_rate':
                models[concept_name] = SpeechRatePredictor(concept_config)
            elif concept_name == 'pause_ratio':
                models[concept_name] = PauseRatioPredictor(concept_config)
            elif concept_name == 'lexical_richness':
                models[concept_name] = LexicalRichnessPredictor(concept_config)
            elif concept_name == 'syntactic_complexity':
                models[concept_name] = SyntacticComplexityPredictor(concept_config)
            else:
                logger.warning(f"未知语音概念: {concept_name}")
                models[concept_name] = self._create_generic_predictor(concept_config)
        
        # EEG概念模型
        for concept_config in self.concept_configs['eeg_concepts']:
            concept_name = concept_config['name']
            
            if concept_name == 'alpha_power':
                models[concept_name] = AlphaPowerPredictor(concept_config)
            elif concept_name == 'theta_beta_ratio':
                models[concept_name] = ThetaBetaRatioPredictor(concept_config)
            else:
                logger.warning(f"未知EEG概念: {concept_name}")
                models[concept_name] = self._create_generic_predictor(concept_config)
        
        return models
    
    def _create_generic_predictor(self, concept_config: Dict) -> nn.Module:
        """创建通用概念预测器"""
        input_dim = 512  # 默认输入维度
        hidden_dims = concept_config.get('hidden_dims', [256, 128])
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, 1))
        
        return nn.Sequential(*layers)
    
    def normalize_by_language(self, concepts: Dict[str, torch.Tensor], language_family: str = 'default') -> Dict[str, torch.Tensor]:
        """根据语言系统特定的标准化参数进行归一化"""
        normalized_concepts = {}
        
        # 获取语言系统特定的归一化参数
        if language_family in self.language_specific_concepts:
            norms = self.language_specific_concepts[language_family]['norms']
        else:
            # 默认使用印欧语系参数
            norms = self.language_specific_concepts['indo_european']['norms']
        
        for concept_name, concept_value in concepts.items():
            if concept_name in self.universal_concepts['acoustic'] + self.universal_concepts['prosodic']:
                # 通用概念不需要语言特定的归一化
                normalized_concepts[concept_name] = concept_value
            elif concept_name in norms:
                # 语言特定概念进行归一化
                mean = norms[concept_name]['mean']
                std = norms[concept_name]['std']
                normalized = (concept_value - mean) / std
                normalized_concepts[concept_name] = normalized
            else:
                normalized_concepts[concept_name] = concept_value
                
        return normalized_concepts
    
    def forward(
        self, 
        features: Dict[str, torch.Tensor],
        language_family: str = 'indo_european',
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """前向传播 - 将底层特征转换为可解释概念
        
        Args:
            features: 多模态特征字典
            language_family: 语言系统标识 ('sino_tibetan' 或 'indo_european')
            return_intermediate: 是否返回中间结果
            
        Returns:
            concepts: 概念预测结果字典
        """
        batch_size = list(features.values())[0].shape[0]
        concepts = {}
        intermediate_outputs = {}
        
        # 1. 处理通用概念
        # 处理声学和韵律特征
        speech_features = features.get('speech', None)
        if speech_features is not None:
            for concept_type in ['acoustic', 'prosodic']:
                for concept_name in self.universal_concepts[concept_type]:
                    if concept_name in self.concept_models:
                        try:
                            concepts[concept_name] = self.concept_models[concept_name](speech_features)
                        except Exception as e:
                            logger.error(f"通用概念 {concept_name} 预测失败: {e}")
                            concepts[concept_name] = torch.zeros(batch_size, 1, device=speech_features.device)
        
        # 处理EEG特征
        eeg_features = features.get('eeg', None)
        if eeg_features is not None:
            for concept_name in self.universal_concepts['eeg']:
                if concept_name in self.concept_models:
                    try:
                        concepts[concept_name] = self.concept_models[concept_name](eeg_features)
                    except Exception as e:
                        logger.error(f"EEG概念 {concept_name} 预测失败: {e}")
                        concepts[concept_name] = torch.zeros(batch_size, 1, device=eeg_features.device)
        
        # 2. 处理语言特定概念
        text_features = features.get('text', None)
        if text_features is not None and language_family in self.language_specific_concepts:
            lang_concepts = self.language_specific_concepts[language_family]
            for concept_type in ['acoustic', 'linguistic']:
                for concept_name in lang_concepts[concept_type]:
                    if concept_name in self.concept_models:
                        try:
                            concepts[concept_name] = self.concept_models[concept_name](
                                speech_features, text_features
                            )
            except Exception as e:
                            logger.error(f"语言特定概念 {concept_name} 预测失败: {e}")
                            concepts[concept_name] = torch.zeros(batch_size, 1, device=text_features.device)
        
        # 3. 应用语言特定的归一化
        concepts = self.normalize_by_language(concepts, language_family)
        
        # 4. 应用概念约束
        concepts = self._apply_concept_constraints(concepts)
        
        if return_intermediate:
            return concepts, intermediate_outputs
        
        return concepts
    
    def _apply_concept_constraints(self, concepts: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """应用概念约束和归一化"""
        constrained_concepts = {}
        
        # 语音概念约束
        for concept_config in self.concept_configs['speech_concepts']:
            concept_name = concept_config['name']
            if concept_name in concepts:
                concept_range = concept_config['range']
                
                # 应用sigmoid激活并缩放到指定范围
                normalized = torch.sigmoid(concepts[concept_name])
                scaled = normalized * (concept_range[1] - concept_range[0]) + concept_range[0]
                constrained_concepts[concept_name] = scaled
        
        # EEG概念约束
        for concept_config in self.concept_configs['eeg_concepts']:
            concept_name = concept_config['name']
            if concept_name in concepts:
                concept_range = concept_config['range']
                
                # 应用ReLU激活并限制到指定范围
                activated = F.relu(concepts[concept_name])
                clamped = torch.clamp(activated, concept_range[0], concept_range[1])
                constrained_concepts[concept_name] = clamped
        
        return constrained_concepts
    
    def _build_diagnostic_model(self) -> nn.Module:
        """构建从概念到诊断的模型(阶段二)"""
        input_dim = len(self.all_concept_names)  # 概念总数
        hidden_dim = 64
        num_classes = 3  # AD/MCI/CN
        
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def compute_loss(
        self, 
        features: Dict[str, torch.Tensor],
        target_concepts: Dict[str, torch.Tensor],
        target_diagnosis: torch.Tensor,
        language_family: str = 'indo_european'
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算总体损失,包含语言特定处理
        
        Args:
            features: 输入特征
            target_concepts: 目标概念值
            target_diagnosis: 目标诊断标签
            language_family: 语言系统('sino_tibetan' 或 'indo_european')
        """
        # 1. 阶段一:特征到概念的预测
        predicted_concepts = self.forward(features, language_family)
        
        # 计算概念预测损失
        concept_loss = 0.0
        individual_concept_losses = {}
        
        # 1.1 处理通用概念损失
        for concept_type in ['acoustic', 'prosodic', 'eeg']:
            for concept_name in self.universal_concepts.get(concept_type, []):
                if concept_name in predicted_concepts and concept_name in target_concepts:
                    pred = predicted_concepts[concept_name]
                    target = target_concepts[concept_name]
                    loss = F.l1_loss(
                        pred,
                        target.unsqueeze(-1) if target.dim() == 1 else target
                    )
                    individual_concept_losses[f"universal_{concept_name}"] = loss
                    concept_loss = concept_loss + loss
        
        # 1.2 处理语言特定概念损失
        if language_family in self.language_specific_concepts:
            lang_concepts = self.language_specific_concepts[language_family]
            for concept_type in ['acoustic', 'linguistic']:
                for concept_name in lang_concepts[concept_type]:
            if concept_name in predicted_concepts and concept_name in target_concepts:
                pred = predicted_concepts[concept_name]
                target = target_concepts[concept_name]
                        loss = F.l1_loss(
                            pred,
                            target.unsqueeze(-1) if target.dim() == 1 else target
                        )
                        individual_concept_losses[f"{language_family}_{concept_name}"] = loss
                        concept_loss = concept_loss + loss
        
        # 2. 阶段二:概念到诊断的预测
        # 将概念组合成向量
        concept_vector = []
        for name in self.all_concept_names:
            if name in predicted_concepts:
                concept_vector.append(predicted_concepts[name])
        concept_vector = torch.cat(concept_vector, dim=-1)
        
        # 预测诊断
        diagnostic_pred = self.diagnostic_model(concept_vector)
        diagnostic_loss = F.cross_entropy(diagnostic_pred, target_diagnosis)
        
        # 3. 计算正则化损失
        regularization_loss = self._compute_regularization_loss(
            predicted_concepts, 
            language_family
        )
        
        # 4. 组合总损失
        total_loss = (
            self.concept_loss_weight * concept_loss +
            self.diagnostic_loss_weight * diagnostic_loss +
            self.concept_regularization * regularization_loss
        )
        
        # 返回损失组成
        loss_components = {
            'concept_loss': concept_loss,
            'diagnostic_loss': diagnostic_loss,
            'regularization_loss': regularization_loss,
            'individual_concept_losses': individual_concept_losses
        }
        
        return total_loss, loss_components
    
    def _compute_regularization_loss(
        self, 
        concepts: Dict[str, torch.Tensor],
        language_family: str
    ) -> torch.Tensor:
        """计算概念正则化损失,考虑语言特定的概念关系"""
        reg_loss = 0.0
        
        # 1. 基础L2正则化
        for concept_value in concepts.values():
            reg_loss = reg_loss + torch.mean(concept_value ** 2)
        
        # 2. 通用概念间的相关性约束
        # 2.1 语音概念关系
        if 'speech_rate' in concepts and 'pause_ratio' in concepts:
            # 语速和停顿比例应该负相关
            correlation_loss = F.relu(
                torch.corrcoef(torch.stack([
                    concepts['speech_rate'].flatten(),
                    concepts['pause_ratio'].flatten()
                ]))[0, 1]
            )
            reg_loss = reg_loss + correlation_loss
            
        # 2.2 EEG概念关系
        if 'alpha_power' in concepts and 'theta_beta_ratio' in concepts:
            correlation_loss = F.relu(
                torch.corrcoef(torch.stack([
                    concepts['alpha_power'].flatten(),
                    concepts['theta_beta_ratio'].flatten()
                ]))[0, 1] - 0.5  # 允许适度正相关
            )
            reg_loss = reg_loss + correlation_loss
        
        # 3. 语言特定的概念关系
        if language_family == 'sino_tibetan':
            # 声调相关特征的约束
            if 'tone_patterns' in concepts and 'tone_errors' in concepts:
                correlation_loss = F.relu(
                    torch.corrcoef(torch.stack([
                        concepts['tone_patterns'].flatten(),
                        concepts['tone_errors'].flatten()
                    ]))[0, 1]
                )
                reg_loss = reg_loss + correlation_loss
                
        elif language_family == 'indo_european':
            # 语法特征的约束
            if 'article_usage' in concepts and 'verb_agreement' in concepts:
                correlation_loss = F.relu(
                    torch.corrcoef(torch.stack([
                        concepts['article_usage'].flatten(),
                        concepts['verb_agreement'].flatten()
                    ]))[0, 1] - 0.4  # 允许中等正相关
                )
                reg_loss = reg_loss + correlation_loss
        
        # 4. 跨模态概念关系
        if 'speech_rate' in concepts and 'alpha_power' in concepts:
            # 语速和脑电活动可能存在关联
            correlation_loss = F.relu(
                abs(torch.corrcoef(torch.stack([
                    concepts['speech_rate'].flatten(),
                    concepts['alpha_power'].flatten()
                ]))[0, 1]) - 0.3  # 允许适度相关
            )
            reg_loss = reg_loss + correlation_loss
        
        return reg_loss
    
    def get_concept_explanations(
        self, 
        concepts: Dict[str, torch.Tensor],
        confidence_threshold: float = 0.5
    ) -> Dict[str, str]:
        """生成概念解释"""
        explanations = {}
        
        for concept_name, concept_value in concepts.items():
            value = concept_value.item() if concept_value.numel() == 1 else concept_value.mean().item()
            explanation = self._generate_concept_explanation(concept_name, value)
            explanations[concept_name] = explanation
        
        return explanations
    
    def _generate_concept_explanation(self, concept_name: str, value: float) -> str:
        """为单个概念生成解释"""
        if concept_name == 'speech_rate':
            if value < 1.5:
                return f"语速较慢 ({value:.2f} 词/秒)，可能表明言语流畅性下降"
            elif value > 2.5:
                return f"语速较快 ({value:.2f} 词/秒)，可能表明焦虑或兴奋状态"
            else:
                return f"语速正常 ({value:.2f} 词/秒)"
        
        elif concept_name == 'pause_ratio':
            if value > 0.3:
                return f"停顿比例较高 ({value:.2f})，可能表明词语查找困难"
            elif value < 0.1:
                return f"停顿比例很低 ({value:.2f})，言语流畅"
            else:
                return f"停顿比例正常 ({value:.2f})"
        
        elif concept_name == 'lexical_richness':
            if value < 0.4:
                return f"词汇丰富度较低 ({value:.2f})，可能表明词汇量减少"
            elif value > 0.7:
                return f"词汇丰富度很高 ({value:.2f})，词汇使用多样"
            else:
                return f"词汇丰富度正常 ({value:.2f})"
        
        elif concept_name == 'syntactic_complexity':
            if value < 3.0:
                return f"句法复杂度较低 ({value:.2f})，句子结构简单"
            elif value > 6.0:
                return f"句法复杂度很高 ({value:.2f})，句子结构复杂"
            else:
                return f"句法复杂度正常 ({value:.2f})"
        
        elif concept_name == 'alpha_power':
            if value < 30.0:
                return f"Alpha波功率较低 ({value:.1f})，可能表明认知活动增强"
            elif value > 70.0:
                return f"Alpha波功率较高 ({value:.1f})，可能表明放松状态"
            else:
                return f"Alpha波功率正常 ({value:.1f})"
        
        elif concept_name == 'theta_beta_ratio':
            if value > 4.0:
                return f"Theta/Beta比值较高 ({value:.2f})，可能表明注意力不集中"
            elif value < 1.0:
                return f"Theta/Beta比值较低 ({value:.2f})，注意力集中"
            else:
                return f"Theta/Beta比值正常 ({value:.2f})"
        
        else:
            return f"{concept_name}: {value:.3f}"
    
    def get_concept_importance(self, concepts: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """计算概念重要性分数"""
        importance_scores = {}
        
        # 基于概念值的偏离程度计算重要性
        for concept_name, concept_value in concepts.items():
            # 获取概念的正常范围
            concept_config = None
            for config in self.concept_configs['speech_concepts'] + self.concept_configs['eeg_concepts']:
                if config['name'] == concept_name:
                    concept_config = config
                    break
            
            if concept_config:
                concept_range = concept_config['range']
                normal_center = (concept_range[0] + concept_range[1]) / 2
                range_size = concept_range[1] - concept_range[0]
                
                value = concept_value.item() if concept_value.numel() == 1 else concept_value.mean().item()
                deviation = abs(value - normal_center) / (range_size / 2)
                importance_scores[concept_name] = min(deviation, 1.0)
            else:
                importance_scores[concept_name] = 0.5
        
        return importance_scores


class ConceptExtractor:
    """概念提取器 - 用于从已有特征中直接提取概念"""
    
    def __init__(self):
        self.speech_concepts = [
            'speech_rate', 'pause_ratio', 'lexical_richness', 'syntactic_complexity'
        ]
        self.eeg_concepts = [
            'alpha_power', 'theta_beta_ratio', 'gamma_connectivity'
        ]
    
    def extract_speech_concepts(
        self, 
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        text: Optional[str] = None
    ) -> Dict[str, float]:
        """从语音和文本特征中直接提取概念"""
        concepts = {}
        
        try:
            # 这里使用传统方法直接计算概念值
            # 实际应用中可能需要训练好的模型
            
            # 语速 (需要文本信息)
            if text:
                word_count = len(text.split())
                # 假设音频特征第一维是时间，每个时间步对应某个时间单位
                duration_estimate = audio_features.shape[0] * 0.01  # 假设每个特征对应10ms
                concepts['speech_rate'] = word_count / max(duration_estimate, 1.0)
            else:
                concepts['speech_rate'] = 0.0
            
            # 停顿比例 (基于能量特征)
            if audio_features.shape[-1] > 0:
                # 假设第一个特征是能量相关
                energy = audio_features[:, 0] if audio_features.dim() > 1 else audio_features
                low_energy_ratio = (energy < energy.quantile(0.2)).float().mean().item()
                concepts['pause_ratio'] = low_energy_ratio
            else:
                concepts['pause_ratio'] = 0.0
            
            # 词汇丰富度和句法复杂度需要文本处理
            concepts['lexical_richness'] = 0.5  # 默认值
            concepts['syntactic_complexity'] = 3.0  # 默认值
            
        except Exception as e:
            logger.error(f"语音概念提取失败: {e}")
            for concept in self.speech_concepts:
                concepts[concept] = 0.0
        
        return concepts
    
    def extract_eeg_concepts(self, eeg_features: torch.Tensor) -> Dict[str, float]:
        """从EEG特征中直接提取概念"""
        concepts = {}
        
        try:
            # Alpha波功率 (假设特征已经是频域特征)
            if eeg_features.shape[-1] >= 5:  # 假设有5个频段的特征
                alpha_power = eeg_features[:, :, 2].mean().item()  # 第3个特征对应alpha
                concepts['alpha_power'] = max(0, alpha_power * 100)  # 缩放到合理范围
            else:
                concepts['alpha_power'] = 50.0
            
            # Theta/Beta比值
            if eeg_features.shape[-1] >= 5:
                theta_power = eeg_features[:, :, 1].mean().item()  # 第2个特征对应theta
                beta_power = eeg_features[:, :, 3].mean().item()   # 第4个特征对应beta
                
                if beta_power > 0:
                    concepts['theta_beta_ratio'] = theta_power / beta_power
                else:
                    concepts['theta_beta_ratio'] = 2.0
            else:
                concepts['theta_beta_ratio'] = 2.0
            
            # Gamma连接性 (简化计算)
            if eeg_features.shape[-1] >= 5:
                gamma_connectivity = eeg_features[:, :, 4].std().item()  # 使用标准差表示连接性
                concepts['gamma_connectivity'] = min(gamma_connectivity, 1.0)
            else:
                concepts['gamma_connectivity'] = 0.5
                
        except Exception as e:
            logger.error(f"EEG概念提取失败: {e}")
            for concept in self.eeg_concepts:
                concepts[concept] = 0.0
        
        return concepts 