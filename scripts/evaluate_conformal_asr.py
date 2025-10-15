#!/usr/bin/env python3
"""
Conformal ASR评估脚本

对比使用和未使用Conformal Inference的ASR识别准确率
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.asr.conformal_enhanced_asr import ConformalEnhancedASR
from src.models.conformal.conformal_asr import ConformalASR
import librosa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ASRComparator:
    """ASR对比评估器"""
    
    def __init__(
        self,
        model_config: Dict,
        output_dir: str = "experiments/conformal_evaluation"
    ):
        self.config = model_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建增强ASR模型
        self.enhanced_asr = ConformalEnhancedASR(model_config)
        
        logger.info("ASR对比评估器初始化完成")
        
    def load_ground_truth(
        self,
        subtitle_dir: str = "data/raw/audio/result"
    ) -> Dict[str, str]:
        """
        加载字幕文件作为真实标注
        
        从result/目录加载.txt字幕文件
        """
        subtitle_dir = Path(subtitle_dir)
        ground_truth = {}
        
        if not subtitle_dir.exists():
            logger.warning(f"字幕目录不存在: {subtitle_dir}")
            return ground_truth
        
        # 读取所有.txt文件
        txt_files = list(subtitle_dir.glob("*.txt"))
        logger.info(f"找到 {len(txt_files)} 个字幕文件")
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                # 提取文件名（不含扩展名）
                file_id = txt_file.stem
                
                # 清理文本（移除时间戳等）
                # 假设字幕格式可能包含时间戳，我们只保留文本内容
                lines = content.split('\n')
                text_lines = []
                for line in lines:
                    # 跳过数字行和时间戳行
                    if line.strip() and not line.strip().isdigit() and '-->' not in line:
                        text_lines.append(line.strip())
                
                text = ' '.join(text_lines)
                ground_truth[file_id] = text
                
            except Exception as e:
                logger.error(f"读取字幕文件失败 {txt_file}: {e}")
        
        logger.info(f"加载了 {len(ground_truth)} 条真实标注")
        return ground_truth
    
    def compute_wer(self, reference: str, hypothesis: str) -> float:
        """
        计算词错误率 (Word Error Rate)
        
        对于中文，按字符计算
        """
        ref_chars = list(reference.replace(" ", ""))
        hyp_chars = list(hypothesis.replace(" ", ""))
        
        # 计算编辑距离
        m, n = len(ref_chars), len(hyp_chars)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_chars[i-1] == hyp_chars[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,      # 删除
                        dp[i][j-1] + 1,      # 插入
                        dp[i-1][j-1] + 1     # 替换
                    )
        
        edit_distance = dp[m][n]
        wer = edit_distance / max(m, 1)
        
        return wer
    
    def compute_accuracy(self, reference: str, hypothesis: str) -> float:
        """计算准确率（1 - WER）"""
        wer = self.compute_wer(reference, hypothesis)
        return max(0.0, 1.0 - wer)
    
    def evaluate_single_audio(
        self,
        audio_path: str,
        ground_truth: str
    ) -> Dict:
        """评估单个音频文件"""
        
        # 1. 不使用Conformal
        self.enhanced_asr.use_conformal = False
        result_without = self.enhanced_asr.forward(audio_path)
        
        # 2. 使用Conformal
        self.enhanced_asr.use_conformal = True
        result_with = self.enhanced_asr.forward(audio_path)
        
        # 3. 计算指标
        wer_without = self.compute_wer(ground_truth, result_without.text)
        wer_with = self.compute_wer(ground_truth, result_with.text)
        
        acc_without = 1.0 - wer_without
        acc_with = 1.0 - wer_with
        
        # 4. 检查覆盖率
        is_covered = any(
            self.compute_accuracy(ground_truth, pred) > 0.8
            for pred in result_with.prediction_set
        )
        
        return {
            'audio_path': audio_path,
            'ground_truth': ground_truth,
            'prediction_without': result_without.text,
            'prediction_with': result_with.text,
            'prediction_set': result_with.prediction_set,
            'set_size': result_with.set_size,
            'wer_without': wer_without,
            'wer_with': wer_with,
            'accuracy_without': acc_without,
            'accuracy_with': acc_with,
            'improvement': acc_with - acc_without,
            'is_covered': is_covered,
            'conformal_confidence': result_with.conformal_confidence
        }
    
    def calibrate_from_samples(
        self,
        audio_paths: List[str],
        ground_truths: List[str],
        calibration_ratio: float = 0.3
    ):
        """
        从样本中选择部分作为校准集
        
        Args:
            audio_paths: 音频路径列表
            ground_truths: 真实文本列表
            calibration_ratio: 校准集比例
        """
        n_calibration = int(len(audio_paths) * calibration_ratio)
        
        # 随机选择校准样本
        indices = np.random.permutation(len(audio_paths))
        calibration_indices = indices[:n_calibration]
        
        calibration_paths = [audio_paths[i] for i in calibration_indices]
        calibration_texts = [ground_truths[i] for i in calibration_indices]
        
        logger.info(f"使用 {len(calibration_paths)} 个样本进行校准")
        
        # 校准
        self.enhanced_asr.calibrate(calibration_paths, calibration_texts)
        
        # 返回剩余的测试样本索引
        test_indices = indices[n_calibration:]
        return test_indices
    
    def evaluate_dataset(
        self,
        audio_dir: str,
        ground_truth_dict: Dict[str, str],
        calibration_ratio: float = 0.3,
        max_samples: Optional[int] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        评估整个数据集
        
        Args:
            audio_dir: 音频目录
            ground_truth_dict: 真实标注字典
            calibration_ratio: 校准集比例
            max_samples: 最大样本数
            
        Returns:
            (详细结果列表, 汇总统计)
        """
        audio_dir = Path(audio_dir)
        
        # 匹配音频和标注
        audio_paths = []
        ground_truths = []
        
        for audio_file in audio_dir.glob("*.wav"):
            file_id = audio_file.stem
            
            # 尝试多种匹配方式
            # 例如: elderly_video_0001.wav -> test1
            match_id = None
            
            # 直接匹配
            if file_id in ground_truth_dict:
                match_id = file_id
            else:
                # 提取数字匹配
                import re
                numbers = re.findall(r'\d+', file_id)
                if numbers:
                    num = int(numbers[0])
                    test_id = f"test{num}"
                    if test_id in ground_truth_dict:
                        match_id = test_id
            
            if match_id:
                audio_paths.append(str(audio_file))
                ground_truths.append(ground_truth_dict[match_id])
        
        logger.info(f"匹配到 {len(audio_paths)} 个音频-标注对")
        
        if len(audio_paths) == 0:
            logger.error("没有找到匹配的音频和标注")
            return [], {}
        
        # 限制样本数
        if max_samples and len(audio_paths) > max_samples:
            indices = np.random.choice(len(audio_paths), max_samples, replace=False)
            audio_paths = [audio_paths[i] for i in indices]
            ground_truths = [ground_truths[i] for i in indices]
        
        # 校准
        test_indices = self.calibrate_from_samples(
            audio_paths, ground_truths, calibration_ratio
        )
        
        # 评估测试集
        results = []
        
        for idx in tqdm(test_indices, desc="评估ASR"):
            audio_path = audio_paths[idx]
            ground_truth = ground_truths[idx]
            
            try:
                result = self.evaluate_single_audio(audio_path, ground_truth)
                results.append(result)
            except Exception as e:
                logger.error(f"评估失败 {audio_path}: {e}")
        
        # 计算汇总统计
        stats = self.compute_statistics(results)
        
        return results, stats
    
    def compute_statistics(self, results: List[Dict]) -> Dict:
        """计算统计信息"""
        
        if len(results) == 0:
            return {}
        
        acc_without = [r['accuracy_without'] for r in results]
        acc_with = [r['accuracy_with'] for r in results]
        improvements = [r['improvement'] for r in results]
        set_sizes = [r['set_size'] for r in results]
        coverage = [r['is_covered'] for r in results]
        
        stats = {
            'n_samples': len(results),
            'avg_accuracy_without': np.mean(acc_without),
            'avg_accuracy_with': np.mean(acc_with),
            'std_accuracy_without': np.std(acc_without),
            'std_accuracy_with': np.std(acc_with),
            'avg_improvement': np.mean(improvements),
            'median_improvement': np.median(improvements),
            'positive_improvement_rate': (np.array(improvements) > 0).mean(),
            'avg_set_size': np.mean(set_sizes),
            'coverage_rate': np.mean(coverage),
            'avg_wer_without': np.mean([r['wer_without'] for r in results]),
            'avg_wer_with': np.mean([r['wer_with'] for r in results])
        }
        
        return stats
    
    def save_results(
        self,
        results: List[Dict],
        stats: Dict,
        prefix: str = "conformal_asr"
    ):
        """保存评估结果"""
        
        # 保存详细结果
        results_path = self.output_dir / f"{prefix}_detailed_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"详细结果已保存: {results_path}")
        
        # 保存统计信息
        stats_path = self.output_dir / f"{prefix}_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"统计信息已保存: {stats_path}")
        
        # 保存为CSV
        df = pd.DataFrame(results)
        csv_path = self.output_dir / f"{prefix}_results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"CSV结果已保存: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="评估Conformal ASR")
    parser.add_argument('--audio_dir', type=str,
                       default='data/raw/audio/elderly_videos',
                       help='音频目录')
    parser.add_argument('--subtitle_dir', type=str,
                       default='data/raw/audio/result',
                       help='字幕目录（真实标注）')
    parser.add_argument('--output_dir', type=str,
                       default='experiments/conformal_evaluation',
                       help='输出目录')
    parser.add_argument('--model_name', type=str,
                       default='large-v3',
                       help='Whisper模型名称')
    parser.add_argument('--coverage', type=float, default=0.95,
                       help='Conformal覆盖率')
    parser.add_argument('--calibration_ratio', type=float, default=0.3,
                       help='校准集比例')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大样本数')
    
    args = parser.parse_args()
    
    # 模型配置
    config = {
        'model_name': args.model_name,
        'language': 'zh',
        'sample_rate': 16000,
        'conformal_coverage': args.coverage,
        'use_conformal': True,
        'beam_size': 5,
        'feature_dim': 512
    }
    
    # 创建评估器
    comparator = ASRComparator(config, args.output_dir)
    
    # 加载真实标注
    ground_truth = comparator.load_ground_truth(args.subtitle_dir)
    
    if len(ground_truth) == 0:
        logger.error("没有加载到真实标注，退出")
        return
    
    # 评估
    results, stats = comparator.evaluate_dataset(
        audio_dir=args.audio_dir,
        ground_truth_dict=ground_truth,
        calibration_ratio=args.calibration_ratio,
        max_samples=args.max_samples
    )
    
    # 保存结果
    comparator.save_results(results, stats)
    
    # 打印统计
    logger.info("\n" + "="*50)
    logger.info("评估结果汇总:")
    logger.info(f"  样本数: {stats['n_samples']}")
    logger.info(f"  平均准确率 (无Conformal): {stats['avg_accuracy_without']:.2%}")
    logger.info(f"  平均准确率 (有Conformal): {stats['avg_accuracy_with']:.2%}")
    logger.info(f"  平均提升: {stats['avg_improvement']:.2%}")
    logger.info(f"  提升率（正向改善比例）: {stats['positive_improvement_rate']:.2%}")
    logger.info(f"  平均预测集大小: {stats['avg_set_size']:.2f}")
    logger.info(f"  覆盖率: {stats['coverage_rate']:.2%}")
    logger.info("="*50)


if __name__ == "__main__":
    main()

