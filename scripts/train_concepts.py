#!/usr/bin/env python3
"""
概念瓶颈层训练脚本
Concept bottleneck layer training script for staged training
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.concepts.concept_extractor import ConceptBottleneckLayer
from src.data.dataset import AlzheimerDataset
from src.utils.metrics import compute_concept_metrics

logger = logging.getLogger(__name__)


class ConceptTrainer:
    """概念瓶颈层训练器"""
    
    def __init__(self, config: dict, experiment_name: str):
        self.config = config
        self.experiment_name = experiment_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化wandb
        if config['training'].get('use_wandb', True):
            wandb.init(
                project="alzheimer-concepts",
                name=experiment_name,
                config=config
            )
        
        # 创建输出目录
        self.output_dir = Path(f"experiments/{experiment_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # 初始化概念层
        self.concept_layer = ConceptBottleneckLayer(config['model']).to(self.device)
        logger.info(f"概念层参数数量: {sum(p.numel() for p in self.concept_layer.parameters()):,}")
        
        # 优化器
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # 训练状态
        self.current_epoch = 0
        self.best_concept_loss = float('inf')
        self.patience_counter = 0
        
    def _setup_optimizer(self):
        """设置优化器"""
        optimizer_config = self.config['training']['optimizer']
        
        if optimizer_config['type'] == 'AdamW':
            return optim.AdamW(
                self.concept_layer.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config['weight_decay']
            )
        elif optimizer_config['type'] == 'Adam':
            return optim.Adam(
                self.concept_layer.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config['weight_decay']
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_config['type']}")
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        scheduler_config = self.config['training']['scheduler']
        
        if scheduler_config['type'] == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config['factor'],
                patience=scheduler_config['patience'],
                verbose=True
            )
        else:
            return None
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.concept_layer.train()
        total_loss = 0.0
        concept_losses = {}
        
        progress_bar = tqdm(train_loader, desc=f"Concept Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移动到设备
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # 准备输入特征
            features = self._prepare_features(batch)
            
            # 前向传播
            self.optimizer.zero_grad()
            concept_predictions = self.concept_layer(features)
            
            # 计算概念损失
            if 'concepts' in batch:
                concept_loss, individual_losses = self.concept_layer.compute_concept_loss(
                    concept_predictions, batch['concepts']
                )
                
                # 反向传播
                concept_loss.backward()
                
                # 梯度裁剪
                if self.config['training'].get('grad_clip_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.concept_layer.parameters(), 
                        self.config['training']['grad_clip_norm']
                    )
                
                self.optimizer.step()
                
                # 记录损失
                total_loss += concept_loss.item()
                for loss_name, loss_value in individual_losses.items():
                    if loss_name not in concept_losses:
                        concept_losses[loss_name] = 0.0
                    concept_losses[loss_name] += loss_value.item()
                
                # 更新进度条
                progress_bar.set_postfix({
                    'concept_loss': f"{concept_loss.item():.4f}",
                    'avg_loss': f"{total_loss/(batch_idx+1):.4f}"
                })
                
                # 记录到wandb
                if self.config['training'].get('use_wandb', True) and batch_idx % 50 == 0:
                    log_dict = {
                        'batch_concept_loss': concept_loss.item(),
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'epoch': self.current_epoch
                    }
                    log_dict.update({f'batch_{k}': v for k, v in individual_losses.items()})
                    wandb.log(log_dict)
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        avg_concept_losses = {k: v / len(train_loader) for k, v in concept_losses.items()}
        
        return avg_loss, avg_concept_losses
    
    def validate(self, val_loader):
        """验证概念层"""
        self.concept_layer.eval()
        total_loss = 0.0
        all_concept_predictions = {}
        all_concept_targets = {}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Concept Validation"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                features = self._prepare_features(batch)
                concept_predictions = self.concept_layer(features)
                
                # 计算损失
                if 'concepts' in batch:
                    concept_loss, _ = self.concept_layer.compute_concept_loss(
                        concept_predictions, batch['concepts']
                    )
                    total_loss += concept_loss.item()
                    
                    # 收集预测和目标
                    for concept_name, concept_pred in concept_predictions.items():
                        if concept_name not in all_concept_predictions:
                            all_concept_predictions[concept_name] = []
                            all_concept_targets[concept_name] = []
                        
                        all_concept_predictions[concept_name].extend(concept_pred.cpu().numpy())
                        all_concept_targets[concept_name].extend(
                            batch['concepts'][concept_name].cpu().numpy()
                        )
        
        # 计算指标
        avg_loss = total_loss / len(val_loader)
        concept_metrics = compute_concept_metrics(all_concept_targets, all_concept_predictions)
        
        return avg_loss, concept_metrics
    
    def _prepare_features(self, batch):
        """准备输入特征"""
        features = {}
        
        # 处理各种特征
        if 'text_features' in batch:
            features['text'] = batch['text_features']
        if 'speech_features' in batch:
            features['speech'] = batch['speech_features']
        if 'eeg_features' in batch:
            features['eeg'] = batch['eeg_features']
        
        # 如果没有特征，创建虚拟特征
        if not features:
            batch_size = len(batch.get('diagnosis', [1]))
            features = {
                'text': torch.zeros(batch_size, 768, device=self.device),
                'speech': torch.zeros(batch_size, 1000, 768, device=self.device),
                'eeg': torch.zeros(batch_size, 100, 19, 15, device=self.device)
            }
        
        return features
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'concept_layer_state_dict': self.concept_layer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_concept_loss': self.best_concept_loss,
            'config': self.config
        }
        
        # 保存最新检查点
        checkpoint_path = self.checkpoint_dir / f"concept_checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.checkpoint_dir / "best_concept_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳概念模型: {best_path}")
    
    def train(self, train_loader, val_loader):
        """主训练循环"""
        logger.info(f"开始概念层训练，总共{self.config['training']['num_epochs']}个epochs")
        
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # 训练
            train_loss, train_concept_losses = self.train_epoch(train_loader)
            
            # 验证
            val_loss, concept_metrics = self.validate(val_loader)
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # 检查是否为最佳模型
            is_best = val_loss < self.best_concept_loss
            if is_best:
                self.best_concept_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 保存检查点
            if epoch % self.config['training']['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # 记录指标
            metrics_log = {
                'epoch': epoch,
                'train_concept_loss': train_loss,
                'val_concept_loss': val_loss,
            }
            
            # 添加概念损失组件
            for loss_name, loss_value in train_concept_losses.items():
                metrics_log[f'train_{loss_name}'] = loss_value
            
            # 添加概念指标
            for concept_name, concept_metric in concept_metrics.items():
                metrics_log[f'{concept_name}_mae'] = concept_metric.get('mae', 0)
                metrics_log[f'{concept_name}_r2'] = concept_metric.get('r2', 0)
            
            # 记录到wandb
            if self.config['training'].get('use_wandb', True):
                wandb.log(metrics_log)
            
            # 打印进度
            logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )
            
            # 早停检查
            if (self.config['training'].get('early_stopping', 0) > 0 and 
                self.patience_counter >= self.config['training']['early_stopping']):
                logger.info(f"概念训练早停触发，在epoch {epoch}")
                break
        
        logger.info("概念层训练完成!")
        return self.best_concept_loss


def main():
    parser = argparse.ArgumentParser(description="概念瓶颈层训练")
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--experiment_name', type=str, required=True, help='实验名称')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 合并数据配置
    data_config_path = Path(args.config).parent / "data_config.yaml"
    if data_config_path.exists():
        with open(data_config_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        config.update(data_config)
    
    # 合并训练配置
    training_config_path = Path(args.config).parent / "training_config.yaml"
    if training_config_path.exists():
        with open(training_config_path, 'r', encoding='utf-8') as f:
            training_config = yaml.safe_load(f)
        config.update(training_config)
    
    # 创建数据集
    logger.info("创建数据集...")
    train_dataset = AlzheimerDataset(
        config['data']['train_data_path'],
        config=config['data'],
        mode='train'
    )
    
    val_dataset = AlzheimerDataset(
        config['data']['val_data_path'],
        config=config['data'],
        mode='val'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # 创建训练器
    trainer = ConceptTrainer(config, args.experiment_name)
    
    # 恢复训练 (如果指定)
    if args.resume:
        logger.info(f"从检查点恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.concept_layer.load_state_dict(checkpoint['concept_layer_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch'] + 1
        trainer.best_concept_loss = checkpoint['best_concept_loss']
    
    # 开始训练
    final_loss = trainer.train(train_loader, val_loader)
    logger.info(f"最终概念损失: {final_loss:.4f}")


if __name__ == "__main__":
    main() 