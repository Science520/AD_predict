import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseConcept(nn.Module):
    """概念预测器基类"""
    
    def __init__(self, concept_config: Dict):
        super().__init__()
        self.concept_config = concept_config
        self.concept_name = concept_config['name']
        self.concept_range = concept_config['range']
        self.model_type = concept_config.get('model_type', 'mlp')
        self.hidden_dims = concept_config.get('hidden_dims', [256, 128])
        
    def _create_mlp(self, input_dim: int, output_dim: int = 1) -> nn.Module:
        """创建多层感知机"""
        layers = []
        current_dim = input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """应用概念特定的激活函数"""
        if self.concept_range[0] >= 0:
            # 非负概念使用ReLU + clamp
            x = F.relu(x)
            x = torch.clamp(x, self.concept_range[0], self.concept_range[1])
        else:
            # 可负概念使用tanh缩放
            x = torch.tanh(x)
            range_center = (self.concept_range[0] + self.concept_range[1]) / 2
            range_scale = (self.concept_range[1] - self.concept_range[0]) / 2
            x = x * range_scale + range_center
        
        return x


class SpeechRatePredictor(BaseConcept):
    """语速预测器
    
    基于语音特征和文本特征预测语速(词/秒)
    """
    
    def __init__(self, concept_config: Dict):
        super().__init__(concept_config)
        
        # 输入维度配置
        self.speech_dim = 768  # 语音特征维度
        self.text_dim = 768    # 文本特征维度
        
        # 特征融合层
        self.speech_projection = nn.Linear(self.speech_dim, 256)
        self.text_projection = nn.Linear(self.text_dim, 256)
        self.fusion_layer = nn.Linear(512, 256)
        
        # 时间注意力机制
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, batch_first=True
        )
        
        # 预测头
        if self.model_type == 'linear':
            self.predictor = nn.Linear(256, 1)
        else:
            self.predictor = self._create_mlp(256)
        
        logger.info(f"初始化语速预测器: {self.model_type}")
    
    def forward(
        self, 
        speech_features: torch.Tensor, 
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            speech_features: 语音特征 [B, T, D_speech]
            text_features: 文本特征 [B, D_text]
            
        Returns:
            speech_rate: 预测的语速 [B, 1]
        """
        batch_size = speech_features.shape[0]
        seq_len = speech_features.shape[1]
        
        # 投影语音特征
        speech_proj = self.speech_projection(speech_features)  # [B, T, 256]
        
        # 投影并扩展文本特征
        text_proj = self.text_projection(text_features)  # [B, 256]
        text_expanded = text_proj.unsqueeze(1).repeat(1, seq_len, 1)  # [B, T, 256]
        
        # 特征融合
        fused_features = torch.cat([speech_proj, text_expanded], dim=-1)  # [B, T, 512]
        fused_features = self.fusion_layer(fused_features)  # [B, T, 256]
        
        # 时间注意力
        attended_features, attention_weights = self.temporal_attention(
            fused_features, fused_features, fused_features
        )  # [B, T, 256]
        
        # 全局池化
        pooled_features = torch.mean(attended_features, dim=1)  # [B, 256]
        
        # 预测语速
        speech_rate = self.predictor(pooled_features)  # [B, 1]
        speech_rate = self._apply_activation(speech_rate)
        
        return speech_rate


class PauseRatioPredictor(BaseConcept):
    """停顿比例预测器
    
    基于语音能量特征预测停顿比例
    """
    
    def __init__(self, concept_config: Dict):
        super().__init__(concept_config)
        
        self.speech_dim = 768
        self.text_dim = 768
        
        # 能量特征提取器
        self.energy_extractor = nn.Conv1d(
            in_channels=self.speech_dim,
            out_channels=128,
            kernel_size=5,
            padding=2
        )
        
        # 停顿检测器
        self.pause_detector = nn.Sequential(
            nn.Linear(128 + 768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        logger.info("初始化停顿比例预测器")
    
    def forward(
        self, 
        speech_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            speech_features: 语音特征 [B, T, D]
            text_features: 文本特征 [B, D]
            
        Returns:
            pause_ratio: 停顿比例 [B, 1]
        """
        # 提取能量特征
        speech_transposed = speech_features.transpose(1, 2)  # [B, D, T]
        energy_features = self.energy_extractor(speech_transposed)  # [B, 128, T]
        energy_features = energy_features.transpose(1, 2)  # [B, T, 128]
        
        # 计算平均能量特征
        avg_energy = torch.mean(energy_features, dim=1)  # [B, 128]
        
        # 结合文本特征
        combined_features = torch.cat([avg_energy, text_features], dim=-1)  # [B, 128+768]
        
        # 预测停顿比例
        pause_ratio = self.pause_detector(combined_features)  # [B, 1]
        pause_ratio = self._apply_activation(pause_ratio)
        
        return pause_ratio


class LexicalRichnessPredictor(BaseConcept):
    """词汇丰富度预测器
    
    主要基于文本特征预测词汇丰富度(TTR等指标)
    """
    
    def __init__(self, concept_config: Dict):
        super().__init__(concept_config)
        
        self.text_dim = 768
        
        # 词汇特征分析器
        self.lexical_analyzer = nn.Sequential(
            nn.Linear(self.text_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        logger.info("初始化词汇丰富度预测器")
    
    def forward(
        self, 
        speech_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            speech_features: 语音特征 [B, T, D] (主要不使用)
            text_features: 文本特征 [B, D]
            
        Returns:
            lexical_richness: 词汇丰富度 [B, 1]
        """
        # 主要使用文本特征
        richness = self.lexical_analyzer(text_features)  # [B, 1]
        richness = self._apply_activation(richness)
        
        return richness


class SyntacticComplexityPredictor(BaseConcept):
    """句法复杂度预测器
    
    基于文本特征预测句法复杂度
    """
    
    def __init__(self, concept_config: Dict):
        super().__init__(concept_config)
        
        self.text_dim = 768
        
        # 句法分析器
        self.syntax_analyzer = nn.Sequential(
            nn.Linear(self.text_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        logger.info("初始化句法复杂度预测器")
    
    def forward(
        self, 
        speech_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            speech_features: 语音特征 [B, T, D] (主要不使用)
            text_features: 文本特征 [B, D]
            
        Returns:
            syntactic_complexity: 句法复杂度 [B, 1]
        """
        complexity = self.syntax_analyzer(text_features)  # [B, 1]
        complexity = self._apply_activation(complexity)
        
        return complexity


class AlphaPowerPredictor(BaseConcept):
    """Alpha波功率预测器
    
    基于EEG特征预测Alpha波功率
    """
    
    def __init__(self, concept_config: Dict):
        super().__init__(concept_config)
        
        self.eeg_dim = 512  # 每个通道的特征维度
        
        # 空间注意力机制 (跨通道)
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=self.eeg_dim, num_heads=8, batch_first=True
        )
        
        # Alpha波特征提取器
        self.alpha_extractor = nn.Sequential(
            nn.Linear(self.eeg_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        logger.info("初始化Alpha波功率预测器")
    
    def forward(self, eeg_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_features: EEG特征 [B, T, C, D]
            
        Returns:
            alpha_power: Alpha波功率 [B, 1]
        """
        batch_size, seq_len, n_channels, feature_dim = eeg_features.shape
        
        # 重塑为 [B*T, C, D] 以便处理空间信息
        eeg_reshaped = eeg_features.view(batch_size * seq_len, n_channels, feature_dim)
        
        # 空间注意力 (跨通道)
        attended_eeg, _ = self.spatial_attention(
            eeg_reshaped, eeg_reshaped, eeg_reshaped
        )  # [B*T, C, D]
        
        # 跨通道平均
        channel_averaged = torch.mean(attended_eeg, dim=1)  # [B*T, D]
        
        # 重塑回时间维度
        temporal_features = channel_averaged.view(batch_size, seq_len, feature_dim)
        
        # 时间平均
        temporal_averaged = torch.mean(temporal_features, dim=1)  # [B, D]
        
        # 预测Alpha波功率
        alpha_power = self.alpha_extractor(temporal_averaged)  # [B, 1]
        alpha_power = self._apply_activation(alpha_power)
        
        return alpha_power


class ThetaBetaRatioPredictor(BaseConcept):
    """Theta/Beta比值预测器
    
    基于EEG特征预测Theta/Beta比值
    """
    
    def __init__(self, concept_config: Dict):
        super().__init__(concept_config)
        
        self.eeg_dim = 512
        
        # 分别预测Theta和Beta功率
        self.theta_extractor = nn.Sequential(
            nn.Linear(self.eeg_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.beta_extractor = nn.Sequential(
            nn.Linear(self.eeg_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 空间注意力
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=self.eeg_dim, num_heads=8, batch_first=True
        )
        
        logger.info("初始化Theta/Beta比值预测器")
    
    def forward(self, eeg_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_features: EEG特征 [B, T, C, D]
            
        Returns:
            theta_beta_ratio: Theta/Beta比值 [B, 1]
        """
        batch_size, seq_len, n_channels, feature_dim = eeg_features.shape
        
        # 重塑为 [B*T, C, D]
        eeg_reshaped = eeg_features.view(batch_size * seq_len, n_channels, feature_dim)
        
        # 空间注意力
        attended_eeg, _ = self.spatial_attention(
            eeg_reshaped, eeg_reshaped, eeg_reshaped
        )  # [B*T, C, D]
        
        # 跨通道平均
        channel_averaged = torch.mean(attended_eeg, dim=1)  # [B*T, D]
        
        # 重塑回时间维度并平均
        temporal_features = channel_averaged.view(batch_size, seq_len, feature_dim)
        temporal_averaged = torch.mean(temporal_features, dim=1)  # [B, D]
        
        # 分别预测Theta和Beta功率
        theta_power = F.relu(self.theta_extractor(temporal_averaged))  # [B, 1]
        beta_power = F.relu(self.beta_extractor(temporal_averaged))  # [B, 1]
        
        # 计算比值 (添加小的epsilon避免除零)
        epsilon = 1e-6
        theta_beta_ratio = theta_power / (beta_power + epsilon)
        
        # 应用激活函数
        theta_beta_ratio = self._apply_activation(theta_beta_ratio)
        
        return theta_beta_ratio


class GammaConnectivityPredictor(BaseConcept):
    """Gamma波连接性预测器
    
    基于EEG特征预测Gamma波段的大脑连接性
    """
    
    def __init__(self, concept_config: Dict):
        super().__init__(concept_config)
        
        self.eeg_dim = 512
        self.n_channels = 19  # 标准EEG通道数
        
        # 连接性分析器
        self.connectivity_analyzer = nn.Sequential(
            nn.Linear(self.eeg_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 跨通道连接性计算
        self.channel_correlation = nn.MultiheadAttention(
            embed_dim=self.eeg_dim, num_heads=8, batch_first=True
        )
        
        logger.info("初始化Gamma连接性预测器")
    
    def forward(self, eeg_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_features: EEG特征 [B, T, C, D]
            
        Returns:
            gamma_connectivity: Gamma连接性 [B, 1]
        """
        batch_size, seq_len, n_channels, feature_dim = eeg_features.shape
        
        # 计算跨通道相关性
        connectivity_scores = []
        
        for t in range(seq_len):
            # 取当前时间步的所有通道 [B, C, D]
            current_time = eeg_features[:, t, :, :]
            
            # 计算通道间注意力
            attended, attention_weights = self.channel_correlation(
                current_time, current_time, current_time
            )  # [B, C, D], [B, C, C]
            
            # 使用注意力权重计算连接强度
            # 注意力权重矩阵对角线元素表示自连接，非对角线表示跨连接
            batch_connectivity = []
            for b in range(batch_size):
                # 取上三角部分 (避免重复计算)
                upper_triangle = torch.triu(attention_weights[b], diagonal=1)
                connectivity_strength = torch.mean(upper_triangle[upper_triangle > 0])
                
                if torch.isnan(connectivity_strength):
                    connectivity_strength = torch.tensor(0.0, device=eeg_features.device)
                
                batch_connectivity.append(connectivity_strength)
            
            connectivity_scores.append(torch.stack(batch_connectivity))
        
        # 时间平均
        avg_connectivity = torch.stack(connectivity_scores, dim=1)  # [B, T]
        temporal_connectivity = torch.mean(avg_connectivity, dim=1, keepdim=True)  # [B, 1]
        
        # 应用激活函数
        gamma_connectivity = self._apply_activation(temporal_connectivity)
        
        return gamma_connectivity


# 概念预测器注册表
CONCEPT_PREDICTORS = {
    'speech_rate': SpeechRatePredictor,
    'pause_ratio': PauseRatioPredictor,
    'lexical_richness': LexicalRichnessPredictor,
    'syntactic_complexity': SyntacticComplexityPredictor,
    'alpha_power': AlphaPowerPredictor,
    'theta_beta_ratio': ThetaBetaRatioPredictor,
    'gamma_connectivity': GammaConnectivityPredictor,
}


def get_concept_predictor(concept_name: str, concept_config: Dict) -> BaseConcept:
    """根据概念名称获取对应的预测器"""
    if concept_name in CONCEPT_PREDICTORS:
        return CONCEPT_PREDICTORS[concept_name](concept_config)
    else:
        logger.warning(f"未知概念: {concept_name}，使用通用预测器")
        # 返回通用预测器
        class GenericPredictor(BaseConcept):
            def __init__(self, config):
                super().__init__(config)
                self.predictor = self._create_mlp(512)
            
            def forward(self, *args):
                # 简单地使用第一个输入
                if len(args) == 1:
                    features = args[0]
                else:
                    features = args[1]  # 使用text features
                
                if features.dim() > 2:
                    features = torch.mean(features, dim=1)
                
                return self._apply_activation(self.predictor(features))
        
        return GenericPredictor(concept_config) 