#!/usr/bin/env python3
"""
GMLVQ (Generalized Matrix Learning Vector Quantization) 患者分层器

基于声学特征对患者进行AI引导的分层，解决异质性问题
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StratificationResult:
    """分层结果"""
    patient_id: str
    stratum: int  # 分层编号
    stratum_name: str  # 分层名称 (e.g., "快速进展者", "稳定者")
    distance_to_prototype: float  # 到原型向量的距离
    confidence: float  # 分层置信度
    features: np.ndarray  # 输入特征


class GMLVQ(nn.Module):
    """
    广义度量学习向量量化模型
    
    通过学习原型向量和度量矩阵，实现患者的精准分层
    """
    
    def __init__(
        self, 
        n_features: int,
        n_prototypes_per_class: int = 1,
        n_classes: int = 3,  # 健康、MCI、AD
        learning_rate: float = 0.01,
        metric_learning_rate: float = 0.001
    ):
        super().__init__()
        
        self.n_features = n_features
        self.n_prototypes_per_class = n_prototypes_per_class
        self.n_classes = n_classes
        self.n_prototypes = n_prototypes_per_class * n_classes
        
        # 原型向量 - 每个类别的代表性样本
        self.prototypes = nn.Parameter(
            torch.randn(self.n_prototypes, n_features) * 0.1
        )
        
        # 原型标签
        self.prototype_labels = torch.repeat_interleave(
            torch.arange(n_classes),
            n_prototypes_per_class
        )
        
        # 度量矩阵 (用于学习特征空间的最优度量)
        # 使用对角矩阵的平方根表示，确保半正定
        self.omega = nn.Parameter(torch.eye(n_features))
        
        self.learning_rate = learning_rate
        self.metric_learning_rate = metric_learning_rate
        
    def compute_distance(self, x: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        计算样本到原型的马氏距离
        
        Args:
            x: [batch_size, n_features]
            prototypes: [n_prototypes, n_features]
            
        Returns:
            distances: [batch_size, n_prototypes]
        """
        # 计算度量矩阵 Lambda = Omega^T * Omega
        lambda_matrix = torch.mm(self.omega.t(), self.omega)
        
        # 计算马氏距离
        batch_size = x.shape[0]
        distances = torch.zeros(batch_size, prototypes.shape[0])
        
        for i in range(batch_size):
            diff = x[i:i+1] - prototypes  # [n_prototypes, n_features]
            # d = (x-p)^T * Lambda * (x-p)
            dist = torch.sum(
                torch.mm(diff, lambda_matrix) * diff,
                dim=1
            )
            distances[i] = dist
            
        return distances
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: [batch_size, n_features]
            
        Returns:
            distances: [batch_size, n_prototypes]
            predictions: [batch_size] 预测的类别
        """
        distances = self.compute_distance(x, self.prototypes)
        
        # 找到每个类别中最近的原型
        class_distances = []
        for c in range(self.n_classes):
            mask = self.prototype_labels == c
            class_dist = distances[:, mask].min(dim=1)[0]
            class_distances.append(class_dist)
            
        class_distances = torch.stack(class_distances, dim=1)  # [batch_size, n_classes]
        
        # 预测为距离最小的类别
        predictions = class_distances.argmin(dim=1)
        
        return distances, predictions
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        计算GMLVQ损失函数
        
        使用广义学习向量量化的损失：
        最小化到正确类别原型的距离，最大化到错误类别原型的距离
        """
        distances = self.compute_distance(x, self.prototypes)
        
        batch_size = x.shape[0]
        loss = 0.0
        
        for i in range(batch_size):
            # 找到正确类别中最近的原型
            correct_mask = self.prototype_labels == y[i]
            d_plus = distances[i, correct_mask].min()
            
            # 找到错误类别中最近的原型
            incorrect_mask = self.prototype_labels != y[i]
            d_minus = distances[i, incorrect_mask].min()
            
            # GMLVQ损失: (d+ - d-) / (d+ + d-)
            # 使用sigmoid激活使其在[0,1]范围内
            mu = (d_plus - d_minus) / (d_plus + d_minus + 1e-10)
            loss += torch.sigmoid(mu)
            
        return loss / batch_size
    
    def predict_with_confidence(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测类别并返回置信度
        
        置信度基于到最近原型和次近原型的距离比
        """
        distances, predictions = self.forward(x)
        
        # 计算置信度
        class_distances = []
        for c in range(self.n_classes):
            mask = self.prototype_labels == c
            class_dist = distances[:, mask].min(dim=1)[0]
            class_distances.append(class_dist)
            
        class_distances = torch.stack(class_distances, dim=1)
        
        # 获取最小和次小距离
        sorted_distances, _ = torch.sort(class_distances, dim=1)
        d_min = sorted_distances[:, 0]
        d_second = sorted_distances[:, 1]
        
        # 置信度: (d_second - d_min) / d_second
        # 距离差异越大，置信度越高
        confidence = (d_second - d_min) / (d_second + 1e-10)
        
        return predictions, confidence


class GMLVQStratifier:
    """
    GMLVQ患者分层器
    
    使用GMLVQ算法对患者进行分层，解决异质性问题
    """
    
    def __init__(
        self,
        n_features: int = 20,  # 声学特征维度
        n_strata: int = 3,  # 分层数量（快速进展者、中等、稳定者）
        n_prototypes_per_stratum: int = 2,
        config: Optional[Dict] = None
    ):
        self.n_features = n_features
        self.n_strata = n_strata
        self.n_prototypes_per_stratum = n_prototypes_per_stratum
        
        self.config = config or {}
        
        # 创建GMLVQ模型
        self.model = GMLVQ(
            n_features=n_features,
            n_prototypes_per_class=n_prototypes_per_stratum,
            n_classes=n_strata,
            learning_rate=self.config.get('learning_rate', 0.01),
            metric_learning_rate=self.config.get('metric_learning_rate', 0.001)
        )
        
        # 特征标准化
        self.scaler = StandardScaler()
        
        # 分层名称
        self.stratum_names = self.config.get('stratum_names', [
            "快速进展者",
            "中等进展者", 
            "稳定者"
        ])
        
        self.is_fitted = False
        
    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True
    ):
        """
        训练GMLVQ模型
        
        Args:
            features: [n_samples, n_features] 声学特征
            labels: [n_samples] 患者标签 (0: 健康, 1: MCI, 2: AD)
            n_epochs: 训练轮数
            batch_size: 批大小
            verbose: 是否显示训练进度
        """
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        
        # 转换为tensor
        X = torch.FloatTensor(features_scaled)
        y = torch.LongTensor(labels)
        
        # 优化器
        optimizer = torch.optim.Adam([
            {'params': [self.model.prototypes], 'lr': self.model.learning_rate},
            {'params': [self.model.omega], 'lr': self.model.metric_learning_rate}
        ])
        
        # 训练循环
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(n_epochs):
            # 打乱数据
            perm = torch.randperm(n_samples)
            X_shuffled = X[perm]
            y_shuffled = y[perm]
            
            epoch_loss = 0.0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # 前向传播
                optimizer.zero_grad()
                loss = self.model.compute_loss(X_batch, y_batch)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / n_batches
                logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
                
        self.is_fitted = True
        logger.info("GMLVQ训练完成!")
        
    def stratify(
        self,
        features: np.ndarray,
        patient_ids: Optional[List[str]] = None
    ) -> List[StratificationResult]:
        """
        对患者进行分层
        
        Args:
            features: [n_samples, n_features] 声学特征
            patient_ids: 患者ID列表
            
        Returns:
            分层结果列表
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
            
        # 标准化特征
        features_scaled = self.scaler.transform(features)
        X = torch.FloatTensor(features_scaled)
        
        # 预测
        with torch.no_grad():
            predictions, confidence = self.model.predict_with_confidence(X)
            distances = self.model.compute_distance(X, self.model.prototypes)
            
        # 生成结果
        results = []
        for i in range(len(features)):
            stratum = predictions[i].item()
            
            # 找到该分层中最近的原型距离
            stratum_mask = self.model.prototype_labels == stratum
            min_distance = distances[i, stratum_mask].min().item()
            
            result = StratificationResult(
                patient_id=patient_ids[i] if patient_ids else f"patient_{i}",
                stratum=stratum,
                stratum_name=self.stratum_names[stratum],
                distance_to_prototype=min_distance,
                confidence=confidence[i].item(),
                features=features[i]
            )
            results.append(result)
            
        return results
    
    def save(self, save_path: str):
        """保存模型"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存模型参数
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_,
            'config': {
                'n_features': self.n_features,
                'n_strata': self.n_strata,
                'n_prototypes_per_stratum': self.n_prototypes_per_stratum,
                'stratum_names': self.stratum_names
            }
        }, save_path)
        
        logger.info(f"模型已保存到: {save_path}")
        
    def load(self, load_path: str):
        """加载模型"""
        checkpoint = torch.load(load_path)
        
        # 恢复模型
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 恢复scaler
        self.scaler.mean_ = checkpoint['scaler_mean']
        self.scaler.scale_ = checkpoint['scaler_scale']
        
        # 恢复配置
        config = checkpoint['config']
        self.n_features = config['n_features']
        self.n_strata = config['n_strata']
        self.n_prototypes_per_stratum = config['n_prototypes_per_stratum']
        self.stratum_names = config['stratum_names']
        
        self.is_fitted = True
        logger.info(f"模型已从 {load_path} 加载")
        
    def visualize_prototypes(self) -> Dict:
        """
        可视化原型向量
        
        返回每个分层的原型向量信息
        """
        prototypes_info = {}
        
        for stratum in range(self.n_strata):
            mask = self.model.prototype_labels == stratum
            stratum_prototypes = self.model.prototypes[mask].detach().numpy()
            
            prototypes_info[self.stratum_names[stratum]] = {
                'prototypes': stratum_prototypes,
                'n_prototypes': stratum_prototypes.shape[0],
                'feature_means': stratum_prototypes.mean(axis=0),
                'feature_stds': stratum_prototypes.std(axis=0)
            }
            
        return prototypes_info

