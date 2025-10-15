#!/usr/bin/env python3
"""
PMM分层独立测试 - 不依赖完整的src模块
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GMLVQ(nn.Module):
    """GMLVQ模型"""
    
    def __init__(self, n_features, n_prototypes_per_class, n_classes):
        super().__init__()
        self.n_features = n_features
        self.n_prototypes_per_class = n_prototypes_per_class
        self.n_classes = n_classes
        self.n_prototypes = n_prototypes_per_class * n_classes
        
        # 原型向量
        self.prototypes = nn.Parameter(torch.randn(self.n_prototypes, n_features) * 0.1)
        
        # 原型标签
        self.prototype_labels = torch.repeat_interleave(
            torch.arange(n_classes),
            n_prototypes_per_class
        )
        
        # 度量矩阵
        self.omega = nn.Parameter(torch.eye(n_features))
    
    def compute_distance(self, x, prototypes):
        """计算马氏距离"""
        lambda_matrix = torch.mm(self.omega.t(), self.omega)
        
        batch_size = x.shape[0]
        distances = torch.zeros(batch_size, prototypes.shape[0])
        
        for i in range(batch_size):
            diff = x[i:i+1] - prototypes
            dist = torch.sum(torch.mm(diff, lambda_matrix) * diff, dim=1)
            distances[i] = dist
        
        return distances
    
    def forward(self, x):
        """前向传播"""
        distances = self.compute_distance(x, self.prototypes)
        
        class_distances = []
        for c in range(self.n_classes):
            mask = self.prototype_labels == c
            class_dist = distances[:, mask].min(dim=1)[0]
            class_distances.append(class_dist)
        
        class_distances = torch.stack(class_distances, dim=1)
        predictions = class_distances.argmin(dim=1)
        
        return distances, predictions
    
    def compute_loss(self, x, y):
        """计算损失"""
        distances = self.compute_distance(x, self.prototypes)
        
        batch_size = x.shape[0]
        loss = 0.0
        
        for i in range(batch_size):
            correct_mask = self.prototype_labels == y[i]
            d_plus = distances[i, correct_mask].min()
            
            incorrect_mask = self.prototype_labels != y[i]
            d_minus = distances[i, incorrect_mask].min()
            
            mu = (d_plus - d_minus) / (d_plus + d_minus + 1e-10)
            loss += torch.sigmoid(mu)
        
        return loss / batch_size


def generate_synthetic_data(n_samples=150):
    """生成合成数据"""
    np.random.seed(42)
    
    n_per_class = n_samples // 3
    
    # 健康组
    healthy_features = np.random.randn(n_per_class, 18) * 0.5
    healthy_features[:, 0] += 200  # F0均值
    healthy_features[:, 8] += 5    # 停顿少
    healthy_labels = np.zeros(n_per_class, dtype=int)
    
    # MCI组
    mci_features = np.random.randn(n_per_class, 18) * 0.8
    mci_features[:, 0] += 180
    mci_features[:, 8] += 15
    mci_labels = np.ones(n_per_class, dtype=int)
    
    # AD组
    ad_features = np.random.randn(n_per_class, 18) * 1.0
    ad_features[:, 0] += 160
    ad_features[:, 8] += 30
    ad_labels = np.full(n_per_class, 2, dtype=int)
    
    # 合并
    features = np.vstack([healthy_features, mci_features, ad_features])
    labels = np.concatenate([healthy_labels, mci_labels, ad_labels])
    
    # 打乱
    shuffle_idx = np.random.permutation(n_samples)
    features = features[shuffle_idx]
    labels = labels[shuffle_idx]
    
    return features, labels


def main():
    logger.info("="*60)
    logger.info("PMM患者分层测试（GMLVQ）")
    logger.info("="*60)
    
    # 1. 生成数据
    logger.info("\n1. 生成合成数据...")
    features, labels = generate_synthetic_data(n_samples=150)
    logger.info(f"数据维度: {features.shape}")
    logger.info(f"类别分布: {np.bincount(labels)}")
    
    # 2. 分割数据
    train_size = int(0.7 * len(features))
    train_features = features[:train_size]
    train_labels = labels[:train_size]
    test_features = features[train_size:]
    test_labels = labels[train_size:]
    
    logger.info(f"训练集: {len(train_features)} 样本")
    logger.info(f"测试集: {len(test_features)} 样本")
    
    # 3. 标准化
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # 转换为tensor
    X_train = torch.FloatTensor(train_features_scaled)
    y_train = torch.LongTensor(train_labels)
    X_test = torch.FloatTensor(test_features_scaled)
    y_test = torch.LongTensor(test_labels)
    
    # 4. 创建模型
    logger.info("\n2. 创建GMLVQ模型...")
    model = GMLVQ(
        n_features=18,
        n_prototypes_per_class=2,
        n_classes=3
    )
    
    # 5. 训练
    logger.info("\n3. 训练模型...")
    optimizer = torch.optim.Adam([
        {'params': [model.prototypes], 'lr': 0.01},
        {'params': [model.omega], 'lr': 0.001}
    ])
    
    n_epochs = 100
    batch_size = 32
    n_batches = (len(X_train) + batch_size - 1) // batch_size
    
    for epoch in range(n_epochs):
        perm = torch.randperm(len(X_train))
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        
        epoch_loss = 0.0
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_train))
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            optimizer.zero_grad()
            loss = model.compute_loss(X_batch, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / n_batches
            logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    logger.info("✅ 训练完成!")
    
    # 6. 测试
    logger.info("\n4. 评估测试集...")
    with torch.no_grad():
        _, predictions = model.forward(X_test)
    
    predictions_np = predictions.numpy()
    test_labels_np = y_test.numpy()
    
    accuracy = (predictions_np == test_labels_np).mean()
    logger.info(f"分层准确率: {accuracy:.2%}")
    
    # 7. 混淆矩阵
    logger.info("\n5. 混淆矩阵:")
    cm = confusion_matrix(test_labels_np, predictions_np)
    print(cm)
    
    # 8. 分类报告
    logger.info("\n6. 分类报告:")
    target_names = ['快速进展者', '中等进展者', '稳定者']
    report = classification_report(test_labels_np, predictions_np, target_names=target_names)
    print(report)
    
    # 9. 保存结果
    results_dir = "experiments/pmm_evaluation"
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存为文本
    with open(f"{results_dir}/pmm_results.txt", "w") as f:
        f.write("PMM患者分层测试结果\n")
        f.write("="*60 + "\n\n")
        f.write(f"数据维度: {features.shape}\n")
        f.write(f"训练集大小: {len(train_features)}\n")
        f.write(f"测试集大小: {len(test_features)}\n\n")
        f.write(f"分层准确率: {accuracy:.2%}\n\n")
        f.write("混淆矩阵:\n")
        f.write(str(cm) + "\n\n")
        f.write("分类报告:\n")
        f.write(report)
    
    logger.info(f"\n结果已保存到: {results_dir}/pmm_results.txt")
    
    logger.info("\n" + "="*60)
    logger.info("✅ PMM分层测试完成!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

