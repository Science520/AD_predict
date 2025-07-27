#!/usr/bin/env python3
"""
模型评估脚本
Model evaluation script for comprehensive assessment
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.integrated_model import IntegratedAlzheimerModel
from src.data.dataset import AlzheimerDataset
from src.utils.metrics import compute_classification_metrics, compute_concept_metrics
from src.utils.visualization import plot_confusion_matrix, plot_concept_predictions

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_path: str, config_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 加载模型
        logger.info(f"从 {model_path} 加载模型...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = IntegratedAlzheimerModel(checkpoint['config']['model'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 类别名称
        self.class_names = checkpoint['config']['model']['crf']['class_names']
        
        logger.info("模型加载完成")
    
    def evaluate_dataset(
        self, 
        dataloader: DataLoader, 
        return_predictions: bool = True
    ) -> dict:
        """评估数据集"""
        
        all_predictions = []
        all_targets = []
        all_concept_predictions = {}
        all_concept_targets = {}
        all_probabilities = []
        sample_results = []
        
        logger.info("开始评估...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(batch, return_concepts=True, return_explanations=True)
                
                # 收集分类结果
                predictions = outputs['diagnosis_predictions'].cpu().numpy()
                probabilities = outputs['diagnosis_probs'].cpu().numpy()
                targets = batch['diagnosis'].cpu().numpy()
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                all_probabilities.extend(probabilities)
                
                # 收集概念预测
                for concept_name, concept_pred in outputs['concepts'].items():
                    if concept_name not in all_concept_predictions:
                        all_concept_predictions[concept_name] = []
                        all_concept_targets[concept_name] = []
                    
                    all_concept_predictions[concept_name].extend(concept_pred.cpu().numpy())
                    
                    if 'concepts' in batch and concept_name in batch['concepts']:
                        all_concept_targets[concept_name].extend(
                            batch['concepts'][concept_name].cpu().numpy()
                        )
                
                # 收集样本级结果（用于详细分析）
                if return_predictions:
                    batch_size = len(predictions)
                    for i in range(batch_size):
                        sample_result = {
                            'prediction': int(predictions[i]),
                            'target': int(targets[i]),
                            'probabilities': probabilities[i].tolist(),
                            'concepts': {name: float(values[i]) 
                                       for name, values in outputs['concepts'].items()}
                        }
                        
                        if 'sample_id' in batch:
                            sample_result['sample_id'] = batch['sample_id'][i]
                        else:
                            sample_result['sample_id'] = batch_idx * dataloader.batch_size + i
                        
                        sample_results.append(sample_result)
        
        # 计算分类指标
        classification_metrics = compute_classification_metrics(
            all_targets, all_predictions, self.class_names
        )
        
        # 计算概念指标
        concept_metrics = {}
        if all_concept_targets:
            concept_metrics = compute_concept_metrics(
                all_concept_targets, all_concept_predictions
            )
        
        # 构建结果字典
        results = {
            'classification_metrics': classification_metrics,
            'concept_metrics': concept_metrics,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'concept_predictions': all_concept_predictions,
            'concept_targets': all_concept_targets,
            'sample_results': sample_results if return_predictions else None
        }
        
        return results
    
    def generate_evaluation_report(
        self, 
        results: dict, 
        output_dir: Path,
        split_name: str = "test"
    ):
        """生成评估报告"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存指标JSON
        metrics_file = output_dir / f"{split_name}_metrics.json"
        metrics_summary = {
            'classification_metrics': results['classification_metrics'],
            'concept_metrics': results['concept_metrics']
        }
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"指标已保存到: {metrics_file}")
        
        # 2. 生成混淆矩阵
        cm_file = output_dir / f"{split_name}_confusion_matrix.png"
        plot_confusion_matrix(
            results['targets'], 
            results['predictions'], 
            self.class_names,
            save_path=str(cm_file)
        )
        
        # 3. 生成概念预测图（如果有概念标签）
        if results['concept_targets']:
            concept_plot_file = output_dir / f"{split_name}_concept_predictions.png"
            plot_concept_predictions(
                results['concept_targets'],
                results['concept_predictions'],
                save_path=str(concept_plot_file)
            )
        
        # 4. 生成详细分类报告
        report_file = output_dir / f"{split_name}_classification_report.txt"
        report = classification_report(
            results['targets'], 
            results['predictions'], 
            target_names=self.class_names,
            digits=4
        )
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("分类报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
            f.write("\n\n")
            f.write("详细指标\n")
            f.write("-" * 30 + "\n")
            for metric, value in results['classification_metrics'].items():
                f.write(f"{metric}: {value:.4f}\n")
        
        # 5. 概念指标报告
        if results['concept_metrics']:
            concept_report_file = output_dir / f"{split_name}_concept_report.txt"
            with open(concept_report_file, 'w', encoding='utf-8') as f:
                f.write("概念预测报告\n")
                f.write("=" * 50 + "\n\n")
                
                for concept_name, metrics in results['concept_metrics'].items():
                    f.write(f"{concept_name}:\n")
                    for metric_name, value in metrics.items():
                        f.write(f"  {metric_name}: {value:.4f}\n")
                    f.write("\n")
        
        # 6. 保存样本级预测结果
        if results['sample_results']:
            samples_file = output_dir / f"{split_name}_sample_predictions.json"
            with open(samples_file, 'w', encoding='utf-8') as f:
                json.dump(results['sample_results'], f, ensure_ascii=False, indent=2)
        
        # 7. 生成HTML报告
        self._generate_html_report(results, output_dir / f"{split_name}_report.html", split_name)
        
        logger.info(f"评估报告已生成到: {output_dir}")
    
    def _generate_html_report(self, results: dict, output_file: Path, split_name: str):
        """生成HTML评估报告"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{split_name.upper()} 数据集评估报告</title>
            <style>
                body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
                .section {{ background: #f8f9fa; margin: 15px 0; padding: 15px; border-radius: 8px; }}
                .metric {{ background: white; margin: 5px 0; padding: 10px; border-radius: 5px; border-left: 4px solid #007bff; }}
                .metric-value {{ font-weight: bold; color: #28a745; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 阿尔茨海默症检测模型评估报告</h1>
                <p>数据集: {split_name.upper()}</p>
                <p>评估时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 分类性能指标</h2>
        """
        
        # 添加分类指标
        for metric, value in results['classification_metrics'].items():
            if not metric.endswith('_Healthy') and not metric.endswith('_MCI') and not metric.endswith('_AD'):
                html_content += f"""
                <div class="metric">
                    <span>{metric}:</span> <span class="metric-value">{value:.4f}</span>
                </div>
                """
        
        # 添加每类别指标表格
        html_content += """
                <h3>各类别详细指标</h3>
                <table>
                    <tr><th>类别</th><th>精确率</th><th>召回率</th><th>F1分数</th></tr>
        """
        
        for class_name in self.class_names:
            precision = results['classification_metrics'].get(f'precision_{class_name}', 0)
            recall = results['classification_metrics'].get(f'recall_{class_name}', 0)
            f1 = results['classification_metrics'].get(f'f1_{class_name}', 0)
            
            html_content += f"""
                    <tr>
                        <td>{class_name}</td>
                        <td>{precision:.4f}</td>
                        <td>{recall:.4f}</td>
                        <td>{f1:.4f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        """
        
        # 添加概念指标（如果有）
        if results['concept_metrics']:
            html_content += """
            <div class="section">
                <h2>🔍 概念预测性能</h2>
                <table>
                    <tr><th>概念</th><th>MAE</th><th>RMSE</th><th>R²</th><th>相关系数</th></tr>
            """
            
            for concept_name, metrics in results['concept_metrics'].items():
                html_content += f"""
                    <tr>
                        <td>{concept_name}</td>
                        <td>{metrics.get('mae', 0):.4f}</td>
                        <td>{metrics.get('rmse', 0):.4f}</td>
                        <td>{metrics.get('r2', 0):.4f}</td>
                        <td>{metrics.get('correlation', 0):.4f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </div>
            """
        
        html_content += """
            <div class="section">
                <h2>📈 可视化图表</h2>
                <p>混淆矩阵和概念预测图表已保存为PNG文件，请查看相关文件。</p>
            </div>
            
            <div class="section">
                <h2>⚠️ 说明</h2>
                <p>本报告展示了模型在指定数据集上的性能表现。请结合领域知识和临床经验解释结果。</p>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description="阿尔茨海默症检测模型评估")
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--config_path', type=str, help='配置文件路径')
    parser.add_argument('--data_path', type=str, required=True, help='测试数据路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--split_name', type=str, default='test', help='数据集分割名称')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载进程数')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 确定配置文件路径
    if args.config_path:
        config_path = args.config_path
    else:
        # 尝试从模型目录找配置文件
        model_dir = Path(args.model_path).parent.parent
        config_path = model_dir / "config" / "model_config.yaml"
        if not config_path.exists():
            config_path = "config/model_config.yaml"
    
    # 创建评估器
    evaluator = ModelEvaluator(args.model_path, config_path)
    
    # 加载数据
    logger.info(f"加载数据: {args.data_path}")
    dataset = AlzheimerDataset(
        args.data_path,
        config=evaluator.config.get('data', {}),
        mode='test'
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 运行评估
    logger.info("开始模型评估...")
    results = evaluator.evaluate_dataset(dataloader, return_predictions=True)
    
    # 生成报告
    output_dir = Path(args.output_dir)
    evaluator.generate_evaluation_report(results, output_dir, args.split_name)
    
    # 打印关键指标
    print("\n" + "="*60)
    print(f"{args.split_name.upper()} 数据集评估结果")
    print("="*60)
    print(f"准确率: {results['classification_metrics']['accuracy']:.4f}")
    print(f"宏平均F1: {results['classification_metrics']['f1_macro']:.4f}")
    print(f"微平均F1: {results['classification_metrics']['f1_micro']:.4f}")
    
    if results['concept_metrics']:
        print("\n概念预测性能:")
        for concept_name, metrics in results['concept_metrics'].items():
            print(f"  {concept_name}: MAE={metrics.get('mae', 0):.4f}, R²={metrics.get('r2', 0):.4f}")
    
    print("="*60)
    logger.info("评估完成!")


if __name__ == "__main__":
    import pandas as pd  # 用于时间戳
    main() 