#!/usr/bin/env python3
"""
Conformal ASR独立测试 - 使用真实样本数据
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import logging
import whisper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_wer(reference, hypothesis):
    """计算词错误率"""
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))
    
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
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    wer = dp[m][n] / max(m, 1)
    return wer


class SimpleConformalASR:
    """简化的Conformal ASR"""
    
    def __init__(self, model_name='base', coverage=0.95):
        logger.info(f"加载Whisper模型: {model_name}...")
        self.model = whisper.load_model(model_name)
        self.coverage = coverage
        self.calibrated_threshold = None
        logger.info("✅ 模型加载完成")
    
    def transcribe(self, audio_path, temperature=0.0):
        """转录音频"""
        result = self.model.transcribe(
            audio_path,
            language='zh',
            task='transcribe',
            temperature=temperature,
            verbose=False
        )
        return result['text'], result
    
    def transcribe_with_candidates(self, audio_path):
        """生成多个候选转录"""
        candidates = []
        
        # 主预测
        text, _ = self.transcribe(audio_path, temperature=0.0)
        candidates.append(text)
        
        # 额外候选（不同温度）
        for temp in [0.2, 0.4, 0.6]:
            try:
                text_alt, _ = self.transcribe(audio_path, temperature=temp)
                if text_alt not in candidates:
                    candidates.append(text_alt)
            except:
                pass
        
        return candidates
    
    def calibrate(self, audio_paths, ground_truths):
        """校准"""
        logger.info(f"校准Conformal模型（{len(audio_paths)}个样本）...")
        
        scores = []
        for audio_path, truth in zip(audio_paths, ground_truths):
            candidates = self.transcribe_with_candidates(audio_path)
            
            # 计算非一致性分数（简化版：1 - 最佳匹配的相似度）
            best_wer = min(compute_wer(truth, cand) for cand in candidates)
            score = best_wer
            scores.append(score)
        
        scores = np.array(scores)
        n = len(scores)
        q = min((n + 1) * self.coverage / n, 1.0)  # 确保在[0,1]范围内
        self.calibrated_threshold = np.quantile(scores, q)
        
        logger.info(f"✅ 校准完成，阈值: {self.calibrated_threshold:.4f}")
    
    def predict(self, audio_path):
        """预测（生成预测集）"""
        candidates = self.transcribe_with_candidates(audio_path)
        
        if self.calibrated_threshold is None:
            # 未校准，返回所有候选
            return {
                'main_prediction': candidates[0],
                'prediction_set': candidates,
                'set_size': len(candidates),
                'confidence': 1.0 / len(candidates)
            }
        
        # 根据阈值过滤候选（简化版：保留所有候选）
        return {
            'main_prediction': candidates[0],
            'prediction_set': candidates[:3],  # 最多3个
            'set_size': min(3, len(candidates)),
            'confidence': 1.0 / min(3, len(candidates))
        }


def load_samples():
    """加载样本数据"""
    audio_dir = Path("data/processed/seniortalk_samples/audio")
    text_dir = Path("data/processed/seniortalk_samples/text")
    
    samples = []
    for audio_file in sorted(audio_dir.glob("*.wav")):
        text_file = text_dir / f"{audio_file.stem}.txt"
        if text_file.exists():
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            if not text.startswith('#') and text:
                samples.append({
                    'audio_path': str(audio_file),
                    'ground_truth': text,
                    'id': audio_file.stem
                })
    
    return samples


def main():
    logger.info("="*60)
    logger.info("Conformal ASR测试")
    logger.info("="*60)
    
    # 1. 加载样本
    logger.info("\n1. 加载样本数据...")
    samples = load_samples()
    logger.info(f"找到 {len(samples)} 个样本")
    
    if len(samples) < 3:
        logger.error("样本不足，至少需要3个")
        return
    
    # 2. 分割数据
    n_calibration = max(2, len(samples) // 2)
    calibration_samples = samples[:n_calibration]
    test_samples = samples[n_calibration:]
    
    logger.info(f"校准集: {len(calibration_samples)} 样本")
    logger.info(f"测试集: {len(test_samples)} 样本")
    
    # 3. 创建模型
    logger.info("\n2. 创建Conformal ASR模型...")
    model = SimpleConformalASR(model_name='base', coverage=0.95)
    
    # 4. 校准
    if len(calibration_samples) > 0:
        calibration_audios = [s['audio_path'] for s in calibration_samples]
        calibration_texts = [s['ground_truth'] for s in calibration_samples]
        model.calibrate(calibration_audios, calibration_texts)
    
    # 5. 测试无Conformal
    logger.info("\n3. 测试标准ASR（无Conformal）...")
    results_without = []
    
    for sample in test_samples:
        text, _ = model.transcribe(sample['audio_path'])
        wer = compute_wer(sample['ground_truth'], text)
        
        results_without.append({
            'id': sample['id'],
            'ground_truth': sample['ground_truth'],
            'prediction': text,
            'wer': wer,
            'accuracy': 1.0 - wer
        })
        
        logger.info(f"  {sample['id']}: WER={wer:.2%}")
    
    # 6. 测试Conformal ASR
    logger.info("\n4. 测试Conformal ASR...")
    results_with = []
    
    for sample in test_samples:
        result = model.predict(sample['audio_path'])
        
        # 检查覆盖
        is_covered = any(
            compute_wer(sample['ground_truth'], pred) < 0.3
            for pred in result['prediction_set']
        )
        
        wer = compute_wer(sample['ground_truth'], result['main_prediction'])
        
        results_with.append({
            'id': sample['id'],
            'ground_truth': sample['ground_truth'],
            'prediction': result['main_prediction'],
            'prediction_set': result['prediction_set'],
            'set_size': result['set_size'],
            'wer': wer,
            'accuracy': 1.0 - wer,
            'is_covered': is_covered,
            'confidence': result['confidence']
        })
        
        logger.info(f"  {sample['id']}: WER={wer:.2%}, 集合大小={result['set_size']}")
    
    # 7. 计算统计
    logger.info("\n5. 计算统计结果...")
    
    avg_accuracy_without = np.mean([r['accuracy'] for r in results_without])
    avg_accuracy_with = np.mean([r['accuracy'] for r in results_with])
    avg_wer_without = np.mean([r['wer'] for r in results_without])
    avg_wer_with = np.mean([r['wer'] for r in results_with])
    avg_set_size = np.mean([r['set_size'] for r in results_with])
    coverage_rate = np.mean([r['is_covered'] for r in results_with])
    
    improvement = avg_accuracy_with - avg_accuracy_without
    
    # 8. 打印结果
    logger.info("\n" + "="*60)
    logger.info("评估结果：")
    logger.info("="*60)
    logger.info(f"测试样本数: {len(test_samples)}")
    logger.info(f"\n无Conformal ASR:")
    logger.info(f"  平均准确率: {avg_accuracy_without:.2%}")
    logger.info(f"  平均WER: {avg_wer_without:.2%}")
    logger.info(f"\n使用Conformal ASR:")
    logger.info(f"  平均准确率: {avg_accuracy_with:.2%}")
    logger.info(f"  平均WER: {avg_wer_with:.2%}")
    logger.info(f"  平均预测集大小: {avg_set_size:.2f}")
    logger.info(f"  覆盖率: {coverage_rate:.2%}")
    logger.info(f"\n改善:")
    logger.info(f"  准确率提升: {improvement:+.2%}")
    logger.info(f"  WER降低: {avg_wer_without - avg_wer_with:+.2%}")
    logger.info("="*60)
    
    # 9. 保存结果
    results_dir = Path("experiments/conformal_evaluation")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'calibration_size': len(calibration_samples),
        'test_size': len(test_samples),
        'results_without_conformal': results_without,
        'results_with_conformal': results_with,
        'statistics': {
            'avg_accuracy_without': float(avg_accuracy_without),
            'avg_accuracy_with': float(avg_accuracy_with),
            'avg_wer_without': float(avg_wer_without),
            'avg_wer_with': float(avg_wer_with),
            'avg_set_size': float(avg_set_size),
            'coverage_rate': float(coverage_rate),
            'improvement': float(improvement)
        }
    }
    
    with open(results_dir / 'conformal_asr_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n结果已保存到: {results_dir}/conformal_asr_results.json")
    
    logger.info("\n✅ Conformal ASR测试完成!")


if __name__ == "__main__":
    main()

