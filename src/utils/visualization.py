"""
可视化工具
Visualization utilities for Alzheimer detection system
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float] = None,
    val_accuracies: List[float] = None,
    save_path: Optional[str] = None
):
    """绘制训练曲线"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    if train_accuracies and val_accuracies:
        axes[1].plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
        axes[1].plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].text(0.5, 0.5, 'No Accuracy Data', ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_concept_predictions(
    concept_true: Dict[str, np.ndarray],
    concept_pred: Dict[str, np.ndarray],
    save_path: Optional[str] = None
):
    """绘制概念预测对比图"""
    
    n_concepts = len(concept_true)
    cols = min(3, n_concepts)
    rows = (n_concepts + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_concepts == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (concept_name, true_values) in enumerate(concept_true.items()):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        if concept_name in concept_pred:
            pred_values = concept_pred[concept_name]
            
            # 散点图
            ax.scatter(true_values, pred_values, alpha=0.6)
            
            # 对角线
            min_val = min(np.min(true_values), np.min(pred_values))
            max_val = max(np.max(true_values), np.max(pred_values))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax.set_xlabel(f'True {concept_name}')
            ax.set_ylabel(f'Predicted {concept_name}')
            ax.set_title(f'{concept_name} Prediction')
            ax.grid(True, alpha=0.3)
            
            # 计算R²
            correlation = np.corrcoef(true_values, pred_values)[0, 1] if len(true_values) > 1 else 0
            ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 隐藏空的子图
    for i in range(n_concepts, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str] = None,
    save_path: Optional[str] = None
):
    """绘制混淆矩阵"""
    
    if class_names is None:
        class_names = ["Healthy", "MCI", "AD"]
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def generate_explanation_html(
    concept_predictions: Dict[str, float],
    diagnosis_prediction: Dict[str, Any],
    concept_importances: Dict[str, float] = None,
    output_path: Optional[str] = None
) -> str:
    """生成解释性HTML报告"""
    
    # 概念描述
    concept_descriptions = {
        'speech_rate': '语速 - 反映语言流畅性和认知处理速度',
        'pause_ratio': '停顿比例 - 反映语言规划和执行能力',
        'lexical_richness': '词汇丰富度 - 反映语言表达和词汇记忆能力',
        'syntactic_complexity': '句法复杂度 - 反映语言认知和执行功能',
        'alpha_power': 'Alpha波功率 - 反映注意力和意识清醒状态',
        'theta_beta_ratio': 'Theta/Beta比值 - 反映认知控制和注意力调节',
        'gamma_connectivity': 'Gamma连通性 - 反映信息整合和认知绑定'
    }
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>阿尔茨海默症检测解释报告</title>
        <style>
            body {{ font-family: 'Arial', sans-serif; margin: 20px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
            .section {{ background: #f8f9fa; margin: 15px 0; padding: 15px; border-radius: 8px; }}
            .concept {{ background: white; margin: 10px 0; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }}
            .diagnosis {{ font-size: 24px; font-weight: bold; color: #dc3545; }}
            .confidence {{ font-size: 18px; color: #28a745; }}
            .progress-bar {{ width: 100%; background-color: #e9ecef; border-radius: 10px; overflow: hidden; }}
            .progress-fill {{ height: 20px; background: linear-gradient(90deg, #28a745, #ffc107, #dc3545); transition: width 0.3s; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧠 阿尔茨海默症多模态检测解释报告</h1>
            <p>基于概念瓶颈模型的可解释性诊断</p>
        </div>
        
        <div class="section">
            <h2>📊 诊断结果</h2>
            <p class="diagnosis">预测结果: {diagnosis_prediction.get('class_name', 'Unknown')}</p>
            <p class="confidence">置信度: {diagnosis_prediction.get('confidence', 0):.1%}</p>
            
            <h3>各类别概率分布:</h3>
    """
    
    for class_name, prob in diagnosis_prediction.get('probabilities', {}).items():
        width_percent = prob * 100
        html_content += f"""
            <p>{class_name}: {prob:.1%}</p>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {width_percent}%"></div>
            </div>
        """
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>🔍 医学概念分析</h2>
            <p>以下概念值提供了诊断的医学解释依据:</p>
    """
    
    for concept_name, value in concept_predictions.items():
        description = concept_descriptions.get(concept_name, concept_name)
        importance = concept_importances.get(concept_name, 0) if concept_importances else 0
        
        html_content += f"""
            <div class="concept">
                <h4>{description}</h4>
                <p><strong>概念值:</strong> {value:.3f}</p>
                {f"<p><strong>重要性:</strong> {importance:.3f}</p>" if concept_importances else ""}
            </div>
        """
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>⚠️ 免责声明</h2>
            <p>本报告仅供医学研究和辅助参考使用，不能替代专业医生的临床诊断。
            如有疑虑或需要确诊，请及时咨询专业医疗机构。</p>
        </div>
    </body>
    </html>
    """
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    return html_content


def create_interactive_concept_plot(
    concept_data: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """创建交互式概念可视化"""
    
    fig = make_subplots(
        rows=len(concept_data), cols=1,
        subplot_titles=list(concept_data.keys()),
        vertical_spacing=0.05
    )
    
    for i, (concept_name, values) in enumerate(concept_data.items(), 1):
        fig.add_trace(
            go.Histogram(x=values, name=concept_name, nbinsx=20),
            row=i, col=1
        )
    
    fig.update_layout(
        height=300 * len(concept_data),
        title_text="医学概念分布分析",
        showlegend=False
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig 