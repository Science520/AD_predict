#!/usr/bin/env python3
"""
端到端阿尔茨海默症检测模型训练脚本
End-to-end training for the integrated Alzheimer detection system
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
from sklearn.metrics import classification_report, confusion_matrix

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.integrated_model import IntegratedAlzheimerModel
from src.data.dataset import AlzheimerDataset
from src.utils.metrics import compute_concept_metrics, compute_classification_metrics
from src.utils.visualization import plot_training_curves, generate_explanation_report

logger = logging.getLogger(__name__)


class AlzheimerTrainer:
    """阿尔茨海默症检测模型训练器"""
    
    def __init__(self, config: dict, experiment_name: str):
        self.config = config
        self.experiment_name = experiment_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化wandb
        if config['training'].get('use_wandb', True):
            wandb.init(
                project="alzheimer-detection",
                name=experiment_name,
                config=config
            )
        
        # 创建输出目录
        self.output_dir = Path(f"experiments/{experiment_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # 初始化模型
        self.model = IntegratedAlzheimerModel(config['model']).to(self.device)
        logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 优化器和调度器
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def _setup_optimizer(self):
        """设置优化器"""
        optimizer_config = self.config['training']['optimizer']
        
        if optimizer_config['type'] == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config['weight_decay'],
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_config['type'] == 'Adam':
            return optim.Adam(
                self.model.parameters(),
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
        elif scheduler_config['type'] == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs']
            )
        else:
            return None
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移动到设备
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(batch, return_concepts=True, return_explanations=False)
            
            # 反向传播
            loss = outputs['total_loss']
            loss.backward()
            
            # 梯度裁剪
            if self.config['training'].get('grad_clip_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['grad_clip_norm']
                )
            
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            for loss_name, loss_value in outputs['loss_components'].items():
                if loss_name not in loss_components:
                    loss_components[loss_name] = 0.0
                loss_components[loss_name] += loss_value.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/(batch_idx+1):.4f}"
            })
            
            # 记录到wandb
            if self.config['training'].get('use_wandb', True) and batch_idx % 50 == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': self.current_epoch
                })
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        avg_loss_components = {k: v / len(train_loader) for k, v in loss_components.items()}
        
        return avg_loss, avg_loss_components
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_concept_predictions = {}
        all_concept_targets = {}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(batch, return_concepts=True)
                
                total_loss += outputs['total_loss'].item()
                
                # 收集预测和目标
                all_predictions.extend(outputs['diagnosis_predictions'].cpu().numpy())
                all_targets.extend(batch['diagnosis'].cpu().numpy())
                
                # 收集概念预测
                for concept_name, concept_pred in outputs['concepts'].items():
                    if concept_name not in all_concept_predictions:
                        all_concept_predictions[concept_name] = []
                        all_concept_targets[concept_name] = []
                    
                    all_concept_predictions[concept_name].extend(concept_pred.cpu().numpy())
                    if 'concepts' in batch:
                        all_concept_targets[concept_name].extend(
                            batch['concepts'][concept_name].cpu().numpy()
                        )
        
        # 计算指标
        avg_loss = total_loss / len(val_loader)
        classification_metrics = compute_classification_metrics(all_targets, all_predictions)
        
        # 计算概念指标 (如果有概念标签)
        concept_metrics = {}
        if all_concept_targets:
            concept_metrics = compute_concept_metrics(
                all_concept_targets, all_concept_predictions
            )
        
        return avg_loss, classification_metrics, concept_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # 保存最新检查点
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型: {best_path}")
    
    def train(self, train_loader, val_loader):
        """主训练循环"""
        logger.info(f"开始训练，总共{self.config['training']['num_epochs']}个epochs")
        
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # 训练
            train_loss, train_loss_components = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_metrics, concept_metrics = self.validate(val_loader)
            
            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 检查是否为最佳模型
            is_best = val_metrics['accuracy'] > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_metrics['accuracy']
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 保存检查点
            if epoch % self.config['training']['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # 记录指标
            metrics_log = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_metrics['accuracy'],
                'val_f1': val_metrics['f1_macro'],
                'val_precision': val_metrics['precision_macro'],
                'val_recall': val_metrics['recall_macro']
            }
            
            # 添加损失组件
            for loss_name, loss_value in train_loss_components.items():
                metrics_log[f'train_{loss_name}'] = loss_value
            
            # 添加概念指标
            for concept_name, concept_metric in concept_metrics.items():
                metrics_log[f'concept_{concept_name}_mae'] = concept_metric.get('mae', 0)
                metrics_log[f'concept_{concept_name}_r2'] = concept_metric.get('r2', 0)
            
            # 记录到wandb
            if self.config['training'].get('use_wandb', True):
                wandb.log(metrics_log)
            
            # 打印进度
            logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_metrics['accuracy']:.4f}"
            )
            
            # 早停检查
            if (self.config['training'].get('early_stopping', 0) > 0 and 
                self.patience_counter >= self.config['training']['early_stopping']):
                logger.info(f"早停触发，在epoch {epoch}")
                break
        
        logger.info("训练完成!")
        return self.best_val_accuracy


def main():
    parser = argparse.ArgumentParser(description="端到端阿尔茨海默症检测模型训练")
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
    trainer = AlzheimerTrainer(config, args.experiment_name)
    
    # 恢复训练 (如果指定)
    if args.resume:
        logger.info(f"从检查点恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch'] + 1
        trainer.best_val_accuracy = checkpoint['best_val_accuracy']
        trainer.best_val_loss = checkpoint['best_val_loss']
    
    # 开始训练
    final_accuracy = trainer.train(train_loader, val_loader)
    logger.info(f"最终验证准确率: {final_accuracy:.4f}")


if __name__ == "__main__":
    main() 