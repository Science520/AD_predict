#!/usr/bin/env python3
"""
阿尔茨海默症检测模型推理脚本
Inference script for Alzheimer detection with explanations
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
import torch
import numpy as np
import json
from typing import Dict, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.integrated_model import IntegratedAlzheimerModel
from src.data.audio_processor import AudioProcessor
from src.data.eeg_processor import EEGProcessor
from src.data.text_processor import TextProcessor
from src.utils.visualization import generate_explanation_html

logger = logging.getLogger(__name__)


class AlzheimerPredictor:
    """阿尔茨海默症检测预测器"""
    
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
        
        # 初始化处理器
        self.audio_processor = AudioProcessor(self.config.get('data', {}))
        self.eeg_processor = EEGProcessor(self.config.get('data', {}))
        self.text_processor = TextProcessor(self.config.get('data', {}))
        
        # 类别名称
        self.class_names = checkpoint['config']['model']['crf']['class_names']
        
        logger.info("模型加载完成")
    
    def preprocess_inputs(
        self, 
        audio_path: Optional[str] = None,
        eeg_path: Optional[str] = None,
        text: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """预处理输入数据"""
        inputs = {}
        
        # 处理音频
        if audio_path and Path(audio_path).exists():
            logger.info(f"处理音频文件: {audio_path}")
            audio_features, speech_features = self.audio_processor.process_audio_file(audio_path)
            
            # 转换为张量
            inputs['audio_features'] = torch.tensor(audio_features).unsqueeze(0).to(self.device)
            inputs['speech_features'] = torch.tensor(speech_features).unsqueeze(0).to(self.device)
            
            # 如果没有提供文本，尝试从音频提取
            if not text:
                # 这里可以集成ASR来提取文本
                text = "extracted from audio"  # 占位符
        
        # 处理EEG
        if eeg_path and Path(eeg_path).exists():
            logger.info(f"处理EEG文件: {eeg_path}")
            eeg_features = self.eeg_processor.process_eeg_file(eeg_path)
            inputs['eeg_features'] = torch.tensor(eeg_features).unsqueeze(0).to(self.device)
        
        # 处理文本
        if text:
            logger.info("处理文本输入")
            text_features = self.text_processor.extract_features(text)
            inputs['text_features'] = torch.tensor(text_features).unsqueeze(0).to(self.device)
        
        # 如果没有任何输入，创建虚拟输入
        if not inputs:
            logger.warning("未提供有效输入，使用虚拟数据")
            batch_size = 1
            inputs = {
                'text_features': torch.zeros(batch_size, 768, device=self.device),
                'eeg_features': torch.zeros(batch_size, 100, 19, 15, device=self.device)
            }
        
        return inputs
    
    def predict(
        self,
        audio_path: Optional[str] = None,
        eeg_path: Optional[str] = None,
        text: Optional[str] = None,
        return_explanations: bool = True
    ) -> Dict[str, any]:
        """进行预测"""
        
        # 预处理输入
        inputs = self.preprocess_inputs(audio_path, eeg_path, text)
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model(
                inputs, 
                return_concepts=True, 
                return_explanations=return_explanations
            )
        
        # 处理输出
        diagnosis_probs = outputs['diagnosis_probs'].cpu().numpy()[0]
        predicted_class = int(outputs['diagnosis_predictions'].cpu().numpy()[0])
        predicted_label = self.class_names[predicted_class]
        
        # 概念预测
        concept_predictions = {}
        for concept_name, concept_values in outputs['concepts'].items():
            concept_predictions[concept_name] = float(concept_values.cpu().numpy()[0])
        
        results = {
            'prediction': {
                'class_index': predicted_class,
                'class_name': predicted_label,
                'confidence': float(diagnosis_probs[predicted_class]),
                'probabilities': {
                    name: float(prob) for name, prob in zip(self.class_names, diagnosis_probs)
                }
            },
            'concepts': concept_predictions,
            'input_info': {
                'has_audio': audio_path is not None,
                'has_eeg': eeg_path is not None,
                'has_text': text is not None,
                'audio_path': audio_path,
                'eeg_path': eeg_path,
                'text_length': len(text) if text else 0
            }
        }
        
        # 添加解释 (如果需要)
        if return_explanations and 'explanations' in outputs:
            results['explanations'] = outputs['explanations']
        
        return results
    
    def predict_batch(
        self,
        data_list: List[Dict[str, str]],
        return_explanations: bool = False
    ) -> List[Dict[str, any]]:
        """批量预测"""
        results = []
        
        for data_item in data_list:
            try:
                result = self.predict(
                    audio_path=data_item.get('audio_path'),
                    eeg_path=data_item.get('eeg_path'),
                    text=data_item.get('text'),
                    return_explanations=return_explanations
                )
                result['item_id'] = data_item.get('id', len(results))
                results.append(result)
            except Exception as e:
                logger.error(f"处理项目 {data_item.get('id', len(results))} 时出错: {e}")
                results.append({
                    'item_id': data_item.get('id', len(results)),
                    'error': str(e)
                })
        
        return results
    
    def generate_report(
        self,
        results: Dict[str, any],
        output_path: str,
        include_plots: bool = True
    ):
        """生成详细的预测报告"""
        
        # 创建HTML报告
        html_content = self._create_html_report(results, include_plots)
        
        # 保存HTML文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"报告已保存到: {output_path}")
    
    def _create_html_report(self, results: Dict[str, any], include_plots: bool) -> str:
        """创建HTML报告内容"""
        
        prediction = results['prediction']
        concepts = results['concepts']
        input_info = results['input_info']
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>阿尔茨海默症检测报告</title>
            <style>
                body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 10px; }}
                .result {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .concept {{ background-color: #fff; padding: 10px; margin: 5px 0; border-left: 4px solid #007acc; }}
                .confidence {{ font-size: 24px; font-weight: bold; color: #d9534f; }}
                .concept-value {{ font-weight: bold; color: #5bc0de; }}
                .info {{ color: #666; font-size: 14px; }}
                .chart {{ margin: 20px 0; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 阿尔茨海默症多模态检测报告</h1>
                <p class="info">基于概念瓶颈模型的可解释性诊断系统</p>
            </div>
            
            <div class="result">
                <h2>📊 诊断结果</h2>
                <p><strong>预测分类:</strong> <span style="color: #d9534f; font-size: 20px;">{prediction['class_name']}</span></p>
                <p><strong>置信度:</strong> <span class="confidence">{prediction['confidence']:.2%}</span></p>
                
                <h3>各类别概率:</h3>
                <ul>
        """
        
        for class_name, prob in prediction['probabilities'].items():
            html_content += f"<li><strong>{class_name}:</strong> {prob:.2%}</li>"
        
        html_content += """
                </ul>
            </div>
            
            <div class="result">
                <h2>🔍 医学概念分析</h2>
                <p class="info">以下是从多模态数据中提取的医学相关概念，为诊断提供可解释性支持：</p>
        """
        
        # 概念解释映射
        concept_descriptions = {
            'speech_rate': '语速 - 每秒说话的词数，反映语言流畅性',
            'pause_ratio': '停顿比例 - 语音中停顿时间的占比，反映语言规划能力',
            'lexical_richness': '词汇丰富度 - 词汇多样性指标，反映语言表达能力',
            'syntactic_complexity': '句法复杂度 - 句子结构的复杂程度，反映认知功能',
            'alpha_power': 'Alpha波功率 - 8-13Hz脑电波功率，反映注意力和认知状态',
            'theta_beta_ratio': 'Theta/Beta比值 - 脑电波比值，反映认知控制能力',
            'gamma_connectivity': 'Gamma连通性 - 高频脑电连通性，反映信息整合能力'
        }
        
        for concept_name, value in concepts.items():
            description = concept_descriptions.get(concept_name, concept_name)
            html_content += f"""
                <div class="concept">
                    <p><strong>{description}</strong></p>
                    <p>值: <span class="concept-value">{value:.3f}</span></p>
                </div>
            """
        
        html_content += f"""
            </div>
            
            <div class="result">
                <h2>📋 输入信息</h2>
                <ul>
                    <li><strong>音频文件:</strong> {'✓ 已提供' if input_info['has_audio'] else '✗ 未提供'}</li>
                    <li><strong>EEG文件:</strong> {'✓ 已提供' if input_info['has_eeg'] else '✗ 未提供'}</li>
                    <li><strong>文本输入:</strong> {'✓ 已提供' if input_info['has_text'] else '✗ 未提供'}</li>
                </ul>
                
                {f"<p><strong>音频路径:</strong> {input_info['audio_path']}</p>" if input_info['audio_path'] else ""}
                {f"<p><strong>EEG路径:</strong> {input_info['eeg_path']}</p>" if input_info['eeg_path'] else ""}
                {f"<p><strong>文本长度:</strong> {input_info['text_length']} 字符</p>" if input_info['text_length'] > 0 else ""}
            </div>
            
            <div class="result">
                <h2>⚠️ 重要说明</h2>
                <p class="info">
                    此报告仅供参考，不能替代专业医生的诊断。如有疑虑，请咨询专业医疗人员。
                    本系统基于人工智能技术，可能存在误差，请结合临床表现综合判断。
                </p>
            </div>
            
            <div class="info" style="text-align: center; margin-top: 30px;">
                <p>报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>系统版本: 阿尔茨海默症多模态检测系统 v1.0</p>
            </div>
        </body>
        </html>
        """
        
        return html_content


def main():
    parser = argparse.ArgumentParser(description="阿尔茨海默症检测模型推理")
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--config_path', type=str, help='配置文件路径')
    parser.add_argument('--audio_path', type=str, help='音频文件路径')
    parser.add_argument('--eeg_path', type=str, help='EEG文件路径')
    parser.add_argument('--text', type=str, help='文本输入')
    parser.add_argument('--output_json', type=str, help='JSON结果输出路径')
    parser.add_argument('--output_html', type=str, help='HTML报告输出路径')
    parser.add_argument('--batch_file', type=str, help='批量处理的JSON文件')
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
    
    # 创建预测器
    predictor = AlzheimerPredictor(args.model_path, config_path)
    
    if args.batch_file:
        # 批量处理
        logger.info(f"批量处理文件: {args.batch_file}")
        with open(args.batch_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        
        results = predictor.predict_batch(data_list, return_explanations=True)
        
        # 保存批量结果
        if args.output_json:
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"批量结果已保存到: {args.output_json}")
    
    else:
        # 单个预测
        logger.info("开始单个预测...")
        results = predictor.predict(
            audio_path=args.audio_path,
            eeg_path=args.eeg_path,
            text=args.text,
            return_explanations=True
        )
        
        # 输出结果
        print("\n" + "="*50)
        print("预测结果:")
        print(f"分类: {results['prediction']['class_name']}")
        print(f"置信度: {results['prediction']['confidence']:.2%}")
        print("\n概念预测:")
        for concept, value in results['concepts'].items():
            print(f"  {concept}: {value:.3f}")
        print("="*50)
        
        # 保存JSON结果
        if args.output_json:
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"JSON结果已保存到: {args.output_json}")
        
        # 生成HTML报告
        if args.output_html:
            predictor.generate_report(results, args.output_html)


if __name__ == "__main__":
    import pandas as pd  # 用于时间戳
    main() 