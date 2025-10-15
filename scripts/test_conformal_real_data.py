#!/usr/bin/env python3
"""
使用真实SeniorTalk数据测试Conformal ASR
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import logging
import whisper
from glob import glob

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
        for temp in [0.2, 0.4]:
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
            
            # 计算非一致性分数
            if truth:
                best_wer = min(compute_wer(truth, cand) for cand in candidates)
                score = best_wer
            else:
                score = 1.0
            scores.append(score)
        
        scores = np.array(scores)
        n = len(scores)
        q = min((n + 1) * self.coverage / n, 1.0)
        self.calibrated_threshold = np.quantile(scores, q)
        
        logger.info(f"✅ 校准完成，阈值: {self.calibrated_threshold:.4f}")
    
    def predict(self, audio_path):
        """预测（生成预测集）"""
        candidates = self.transcribe_with_candidates(audio_path)
        
        return {
            'main_prediction': candidates[0],
            'prediction_set': candidates[:3],
            'set_size': min(3, len(candidates)),
            'confidence': 1.0 / min(3, len(candidates))
        }


def main():
    logger.info("="*60)
    logger.info("Conformal ASR真实数据测试")
    logger.info("="*60)
    
    # 1. 查找音频文件
    logger.info("\n1. 查找真实音频数据...")
    audio_dir = Path("data/processed/seniortalk_extracted/S0001")
    audio_files = sorted(glob(str(audio_dir / "*.wav")))
    
    logger.info(f"找到 {len(audio_files)} 个音频文件")
    
    if len(audio_files) < 3:
        logger.error("音频文件不足")
        return
    
    # 2. 分割数据
    n_calibration = max(2, len(audio_files) // 2)
    calibration_files = audio_files[:n_calibration]
    test_files = audio_files[n_calibration:]
    
    logger.info(f"校准集: {len(calibration_files)} 个音频")
    logger.info(f"测试集: {len(test_files)} 个音频")
    
    # 3. 创建模型
    logger.info("\n2. 创建Conformal ASR模型...")
    model = SimpleConformalASR(model_name='base', coverage=0.95)
    
    # 4. 校准（不使用ground truth，仅演示流程）
    logger.info("\n3. 校准模型...")
    calibration_texts = [""] * len(calibration_files)  # 无标注
    model.calibrate(calibration_files, calibration_texts)
    
    # 5. 测试无Conformal
    logger.info("\n4. 测试标准ASR（无Conformal）...")
    results_without = []
    
    for audio_file in test_files:
        text, _ = model.transcribe(audio_file)
        
        results_without.append({
            'file': Path(audio_file).name,
            'prediction': text,
            'length': len(text)
        })
        
        logger.info(f"  {Path(audio_file).name}: {text[:50]}...")
    
    # 6. 测试Conformal ASR
    logger.info("\n5. 测试Conformal ASR...")
    results_with = []
    
    for audio_file in test_files:
        result = model.predict(audio_file)
        
        results_with.append({
            'file': Path(audio_file).name,
            'prediction': result['main_prediction'],
            'prediction_set': result['prediction_set'],
            'set_size': result['set_size'],
            'confidence': result['confidence'],
            'length': len(result['main_prediction'])
        })
        
        logger.info(f"  {Path(audio_file).name}:")
        logger.info(f"    主预测: {result['main_prediction'][:50]}...")
        logger.info(f"    集合大小: {result['set_size']}")
        for i, pred in enumerate(result['prediction_set']):
            logger.info(f"    候选{i+1}: {pred[:50]}...")
    
    # 7. 统计
    logger.info("\n6. 统计结果...")
    
    avg_length_without = np.mean([r['length'] for r in results_without])
    avg_length_with = np.mean([r['length'] for r in results_with])
    avg_set_size = np.mean([r['set_size'] for r in results_with])
    avg_confidence = np.mean([r['confidence'] for r in results_with])
    
    logger.info("\n" + "="*60)
    logger.info("评估结果：")
    logger.info("="*60)
    logger.info(f"测试样本数: {len(test_files)}")
    logger.info(f"\n无Conformal ASR:")
    logger.info(f"  平均文本长度: {avg_length_without:.1f} 字符")
    logger.info(f"\n使用Conformal ASR:")
    logger.info(f"  平均文本长度: {avg_length_with:.1f} 字符")
    logger.info(f"  平均预测集大小: {avg_set_size:.2f}")
    logger.info(f"  平均置信度: {avg_confidence:.3f}")
    logger.info("="*60)
    
    # 8. 保存结果
    results_dir = Path("experiments/conformal_real_data")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'test_size': len(test_files),
        'calibration_size': len(calibration_files),
        'results_without_conformal': results_without,
        'results_with_conformal': results_with,
        'statistics': {
            'avg_length_without': float(avg_length_without),
            'avg_length_with': float(avg_length_with),
            'avg_set_size': float(avg_set_size),
            'avg_confidence': float(avg_confidence)
        }
    }
    
    with open(results_dir / 'conformal_real_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n✅ 结果已保存到: {results_dir}/conformal_real_results.json")
    logger.info("\n✅ Conformal ASR真实数据测试完成!")


if __name__ == "__main__":
    main()

















