import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

from .audio_processor import AudioProcessor
from .eeg_processor import EEGProcessor
from .text_processor import TextProcessor

logger = logging.getLogger(__name__)


class AlzheimerDataset(Dataset):
    """阿尔茨海默症多模态数据集
    
    支持音频、EEG和文本三种模态的数据加载和预处理
    """
    
    def __init__(
        self, 
        data_config: Dict, 
        model_config: Dict,
        mode: str = 'train',
        transform: Optional[Any] = None
    ):
        """
        Args:
            data_config: 数据配置
            model_config: 模型配置  
            mode: 数据集模式 ('train', 'val', 'test')
            transform: 数据变换
        """
        self.data_config = data_config
        self.model_config = model_config
        self.mode = mode
        self.transform = transform
        
        # 初始化处理器
        self.audio_processor = AudioProcessor(data_config['audio'])
        self.eeg_processor = EEGProcessor(data_config['eeg'])
        self.text_processor = TextProcessor(data_config['text'])
        
        # 加载标注数据
        self.data_df = self._load_annotations()
        
        # 标签映射
        self.label_map = {
            'Healthy': 0,
            'MCI': 1, 
            'AD': 2
        }
        
        logger.info(f"加载 {mode} 数据集: {len(self.data_df)} 个样本")
        
    def _load_annotations(self) -> pd.DataFrame:
        """加载标注数据"""
        annotation_path = self.data_config[f'{self.mode}_annotations']
        
        if not Path(annotation_path).exists():
            logger.warning(f"标注文件不存在: {annotation_path}")
            # 创建一个示例数据框
            return self._create_dummy_dataframe()
            
        return pd.read_csv(annotation_path)
    
    def _create_dummy_dataframe(self) -> pd.DataFrame:
        """创建示例数据框(用于测试)"""
        dummy_data = {
            'patient_id': [f'patient_{i:03d}' for i in range(100)],
            'audio_path': [f'data/raw/audio/patient_{i:03d}.wav' for i in range(100)],
            'eeg_path': [f'data/raw/eeg/patient_{i:03d}.edf' for i in range(100)],
            'transcription': [f'This is a sample transcription for patient {i}.' for i in range(100)],
            'diagnosis': np.random.choice(['Healthy', 'MCI', 'AD'], 100),
            'age': np.random.randint(50, 90, 100),
            'gender': np.random.choice(['M', 'F'], 100),
            # 概念标签
            'speech_rate': np.random.uniform(0.5, 4.0, 100),
            'pause_ratio': np.random.uniform(0.0, 1.0, 100),
            'lexical_richness': np.random.uniform(0.0, 1.0, 100),
            'syntactic_complexity': np.random.uniform(1.0, 10.0, 100),
            'alpha_power': np.random.uniform(0.0, 100.0, 100),
            'theta_beta_ratio': np.random.uniform(0.0, 10.0, 100),
        }
        return pd.DataFrame(dummy_data)
    
    def __len__(self) -> int:
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本
        
        Returns:
            Dict包含:
            - audio_features: 音频特征 [T, D]
            - eeg_features: EEG特征 [T, C, D] 
            - text_features: 文本特征 [D]
            - concepts: 概念标签 Dict
            - diagnosis: 诊断标签 int
            - metadata: 元数据
        """
        try:
            row = self.data_df.iloc[idx]
            
            # 加载音频数据
            audio_features = self._load_audio(row['audio_path'])
            
            # 加载EEG数据
            eeg_features = self._load_eeg(row['eeg_path'])
            
            # 处理文本数据
            text_features = self._process_text(row['transcription'])
            
            # 加载概念标签
            concepts = self._load_concept_labels(row)
            
            # 处理诊断标签
            diagnosis = self.label_map[row['diagnosis']]
            
            # 元数据
            metadata = {
                'patient_id': row['patient_id'],
                'age': row.get('age', -1),
                'gender': row.get('gender', 'Unknown'),
                'diagnosis_text': row['diagnosis']
            }
            
            sample = {
                'audio_features': audio_features,
                'eeg_features': eeg_features, 
                'text_features': text_features,
                'concepts': concepts,
                'diagnosis': torch.tensor(diagnosis, dtype=torch.long),
                'metadata': metadata
            }
            
            if self.transform:
                sample = self.transform(sample)
                
            return sample
            
        except Exception as e:
            logger.error(f"加载样本 {idx} 时出错: {e}")
            # 返回一个空样本
            return self._get_empty_sample()
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """加载和预处理音频数据"""
        try:
            if Path(audio_path).exists():
                return self.audio_processor.process(audio_path)
            else:
                logger.warning(f"音频文件不存在: {audio_path}")
                # 返回零填充的特征
                return torch.zeros(1000, self.model_config['asr']['feature_dim'])
        except Exception as e:
            logger.error(f"处理音频文件 {audio_path} 时出错: {e}")
            return torch.zeros(1000, self.model_config['asr']['feature_dim'])
    
    def _load_eeg(self, eeg_path: str) -> torch.Tensor:
        """加载和预处理EEG数据"""
        try:
            if Path(eeg_path).exists():
                return self.eeg_processor.process(eeg_path)
            else:
                logger.warning(f"EEG文件不存在: {eeg_path}")
                # 返回零填充的特征
                n_channels = len(self.data_config['eeg']['channels'])
                feature_dim = self.model_config['eeg']['feature_dim']
                return torch.zeros(100, n_channels, feature_dim)
        except Exception as e:
            logger.error(f"处理EEG文件 {eeg_path} 时出错: {e}")
            n_channels = len(self.data_config['eeg']['channels'])
            feature_dim = self.model_config['eeg']['feature_dim']
            return torch.zeros(100, n_channels, feature_dim)
    
    def _process_text(self, text: str) -> torch.Tensor:
        """处理文本数据"""
        try:
            return self.text_processor.process(text)
        except Exception as e:
            logger.error(f"处理文本时出错: {e}")
            return torch.zeros(self.model_config['text']['feature_dim'])
    
    def _load_concept_labels(self, row: pd.Series) -> Dict[str, torch.Tensor]:
        """加载概念标签"""
        concepts = {}
        
        # 语音概念
        speech_concept_names = [c['name'] for c in self.model_config['concepts']['speech_concepts']]
        for concept_name in speech_concept_names:
            value = row.get(concept_name, 0.0)
            concepts[concept_name] = torch.tensor(float(value), dtype=torch.float32)
        
        # EEG概念
        eeg_concept_names = [c['name'] for c in self.model_config['concepts']['eeg_concepts']]
        for concept_name in eeg_concept_names:
            value = row.get(concept_name, 0.0)
            concepts[concept_name] = torch.tensor(float(value), dtype=torch.float32)
            
        return concepts
    
    def _get_empty_sample(self) -> Dict[str, torch.Tensor]:
        """返回空样本(错误处理)"""
        return {
            'audio_features': torch.zeros(1000, self.model_config['asr']['feature_dim']),
            'eeg_features': torch.zeros(100, len(self.data_config['eeg']['channels']), 
                                      self.model_config['eeg']['feature_dim']),
            'text_features': torch.zeros(self.model_config['text']['feature_dim']),
            'concepts': {name: torch.tensor(0.0) for name in self._get_all_concept_names()},
            'diagnosis': torch.tensor(0, dtype=torch.long),
            'metadata': {'patient_id': 'unknown', 'age': -1, 'gender': 'Unknown', 'diagnosis_text': 'Unknown'}
        }
    
    def _get_all_concept_names(self) -> List[str]:
        """获取所有概念名称"""
        names = []
        for concept in self.model_config['concepts']['speech_concepts']:
            names.append(concept['name'])
        for concept in self.model_config['concepts']['eeg_concepts']:
            names.append(concept['name'])
        return names
    
    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重(用于处理类别不平衡)"""
        label_counts = self.data_df['diagnosis'].value_counts()
        total_samples = len(self.data_df)
        
        weights = []
        for class_name in ['Healthy', 'MCI', 'AD']:
            if class_name in label_counts:
                weight = total_samples / (len(self.label_map) * label_counts[class_name])
            else:
                weight = 1.0
            weights.append(weight)
            
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_concept_statistics(self) -> Dict[str, Dict[str, float]]:
        """获取概念统计信息"""
        stats = {}
        concept_names = self._get_all_concept_names()
        
        for concept_name in concept_names:
            if concept_name in self.data_df.columns:
                values = self.data_df[concept_name].dropna()
                stats[concept_name] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'count': len(values)
                }
            else:
                stats[concept_name] = {
                    'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 1.0, 'count': 0
                }
                
        return stats 