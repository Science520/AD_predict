#!/usr/bin/env python3
"""
评估下载的老年人视频的ASR性能

使用真实字幕作为ground truth，评估Conformal ASR的效果
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
    """计算词错误率（字符级别，适合中文）"""
    ref_chars = list(reference.replace(" ", "").replace("\n", ""))
    hyp_chars = list(hypothesis.replace(" ", "").replace("\n", ""))
    
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


class ElderlyVideoASREvaluator:
    """老年人视频ASR评估器"""
    
    def __init__(self, model_name='base', coverage=0.95):
        logger.info(f"加载Whisper模型: {model_name}...")
        self.model = whisper.load_model(model_name)
        self.coverage = coverage
        self.calibrated_threshold = None
        logger.info("✅ 模型加载完成")
    
    def load_audio_subtitle_pairs(self, audio_dir, subtitle_dir):
        """加载音频和字幕配对"""
        audio_dir = Path(audio_dir)
        subtitle_dir = Path(subtitle_dir)
        
        # 读取音频清单
        manifest_path = audio_dir / 'audio_manifest.json'
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                audio_manifest = json.load(f)
        else:
            logger.warning("未找到audio_manifest.json，扫描音频文件...")
            audio_files = sorted(audio_dir.glob("*.wav"))
            audio_manifest = [
                {'index': i+1, 'audio_path': str(f)}
                for i, f in enumerate(audio_files)
            ]
        
        pairs = []
        
        for item in audio_manifest:
            idx = item['index']
            audio_path = item['audio_path']
            
            # 尝试匹配字幕文件：test{idx}.txt
            subtitle_path = subtitle_dir / f"test{idx}.txt"
            
            if subtitle_path.exists():
                # 读取字幕
                with open(subtitle_path, 'r', encoding='utf-8') as f:
                    subtitle_text = f.read().strip()
                
                pairs.append({
                    'index': idx,
                    'audio_path': audio_path,
                    'subtitle_text': subtitle_text,
                    'subtitle_path': str(subtitle_path)
                })
                logger.info(f"  匹配 #{idx}: {Path(audio_path).name} <-> test{idx}.txt")
            else:
                logger.warning(f"  跳过 #{idx}: 未找到字幕 test{idx}.txt")
        
        logger.info(f"\n成功匹配 {len(pairs)} 对音频-字幕")
        return pairs
    
    def transcribe(self, audio_path, temperature=0.0):
        """转录音频"""
        result = self.model.transcribe(
            audio_path,
            language='zh',
            task='transcribe',
            temperature=temperature,
            verbose=False
        )
        return result['text']
    
    def transcribe_with_candidates(self, audio_path):
        """生成多个候选转录（不同温度）"""
        candidates = []
        
        # 主预测 (temperature=0)
        text = self.transcribe(audio_path, temperature=0.0)
        candidates.append(text)
        
        # 额外候选
        for temp in [0.2, 0.4, 0.6]:
            try:
                text_alt = self.transcribe(audio_path, temperature=temp)
                if text_alt not in candidates:
                    candidates.append(text_alt)
            except:
                pass
        
        return candidates
    
    def calibrate(self, calibration_pairs):
        """使用部分数据校准Conformal模型"""
        logger.info(f"\n校准Conformal模型（{len(calibration_pairs)}个样本）...")
        
        scores = []
        for pair in calibration_pairs:
            audio_path = pair['audio_path']
            ground_truth = pair['subtitle_text']
            
            # 生成候选
            candidates = self.transcribe_with_candidates(audio_path)
            
            # 计算非一致性分数：最佳候选的WER
            best_wer = min(compute_wer(ground_truth, cand) for cand in candidates)
            scores.append(best_wer)
        
        scores = np.array(scores)
        n = len(scores)
        q = min((n + 1) * self.coverage / n, 1.0)
        self.calibrated_threshold = np.quantile(scores, q)
        
        logger.info(f"✅ 校准完成，阈值: {self.calibrated_threshold:.4f}")
    
    def evaluate_pair(self, pair, use_conformal=False):
        """评估单个音频-字幕对"""
        audio_path = pair['audio_path']
        ground_truth = pair['subtitle_text']
        
        if use_conformal:
            # Conformal模式：生成预测集
            candidates = self.transcribe_with_candidates(audio_path)
            main_prediction = candidates[0]
            
            # 根据阈值过滤候选（简化：保留前3个）
            prediction_set = candidates[:min(3, len(candidates))]
            
            # 计算指标
            wer = compute_wer(ground_truth, main_prediction)
            
            # 检查真值是否被覆盖（任一候选WER < 30%）
            is_covered = any(compute_wer(ground_truth, cand) < 0.3 for cand in prediction_set)
            
            return {
                'prediction': main_prediction,
                'prediction_set': prediction_set,
                'set_size': len(prediction_set),
                'wer': wer,
                'accuracy': 1.0 - wer,
                'is_covered': is_covered,
                'confidence': 1.0 / len(prediction_set)
            }
        else:
            # 标准模式：单一预测
            prediction = self.transcribe(audio_path, temperature=0.0)
            wer = compute_wer(ground_truth, prediction)
            
            return {
                'prediction': prediction,
                'wer': wer,
                'accuracy': 1.0 - wer
            }
    
    def evaluate_all(self, pairs, calibration_ratio=0.3):
        """评估所有音频-字幕对"""
        
        # 分割校准集和测试集
        n_calibration = max(1, int(len(pairs) * calibration_ratio))
        calibration_pairs = pairs[:n_calibration]
        test_pairs = pairs[n_calibration:]
        
        logger.info(f"\n数据分割:")
        logger.info(f"  校准集: {len(calibration_pairs)} 样本")
        logger.info(f"  测试集: {len(test_pairs)} 样本")
        
        # 校准
        if len(calibration_pairs) > 0:
            self.calibrate(calibration_pairs)
        
        # 评估测试集 - 无Conformal
        logger.info(f"\n{'='*60}")
        logger.info("评估标准ASR（无Conformal）...")
        logger.info(f"{'='*60}")
        
        results_without = []
        for i, pair in enumerate(test_pairs, 1):
            logger.info(f"\n[{i}/{len(test_pairs)}] {Path(pair['audio_path']).name}")
            result = self.evaluate_pair(pair, use_conformal=False)
            
            logger.info(f"  真实字幕: {pair['subtitle_text'][:80]}...")
            logger.info(f"  ASR识别:  {result['prediction'][:80]}...")
            logger.info(f"  准确率: {result['accuracy']:.2%} (WER: {result['wer']:.2%})")
            
            results_without.append({
                'index': pair['index'],
                'audio_file': Path(pair['audio_path']).name,
                'ground_truth': pair['subtitle_text'],
                **result
            })
        
        # 评估测试集 - 使用Conformal
        logger.info(f"\n{'='*60}")
        logger.info("评估Conformal ASR...")
        logger.info(f"{'='*60}")
        
        results_with = []
        for i, pair in enumerate(test_pairs, 1):
            logger.info(f"\n[{i}/{len(test_pairs)}] {Path(pair['audio_path']).name}")
            result = self.evaluate_pair(pair, use_conformal=True)
            
            logger.info(f"  真实字幕: {pair['subtitle_text'][:80]}...")
            logger.info(f"  主预测:   {result['prediction'][:80]}...")
            logger.info(f"  预测集大小: {result['set_size']}")
            logger.info(f"  准确率: {result['accuracy']:.2%} (WER: {result['wer']:.2%})")
            logger.info(f"  是否覆盖真值: {'✓' if result['is_covered'] else '✗'}")
            
            results_with.append({
                'index': pair['index'],
                'audio_file': Path(pair['audio_path']).name,
                'ground_truth': pair['subtitle_text'],
                **result
            })
        
        return results_without, results_with


def main():
    logger.info("="*60)
    logger.info("老年人视频ASR评估（使用真实字幕）")
    logger.info("="*60)
    
    # 配置
    audio_dir = "data/raw/audio/elderly_audios"
    subtitle_dir = "data/raw/audio/result"
    output_dir = Path("experiments/elderly_video_asr_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建评估器
    evaluator = ElderlyVideoASREvaluator(model_name='base', coverage=0.95)
    
    # 加载音频-字幕配对
    logger.info("\n加载音频-字幕配对...")
    pairs = evaluator.load_audio_subtitle_pairs(audio_dir, subtitle_dir)
    
    if len(pairs) < 2:
        logger.error("配对数据不足（至少需要2对）")
        return
    
    # 评估
    results_without, results_with = evaluator.evaluate_all(pairs, calibration_ratio=0.3)
    
    # 计算统计
    logger.info(f"\n{'='*60}")
    logger.info("评估结果统计")
    logger.info(f"{'='*60}")
    
    avg_acc_without = np.mean([r['accuracy'] for r in results_without])
    avg_acc_with = np.mean([r['accuracy'] for r in results_with])
    avg_wer_without = np.mean([r['wer'] for r in results_without])
    avg_wer_with = np.mean([r['wer'] for r in results_with])
    
    if results_with:
        avg_set_size = np.mean([r['set_size'] for r in results_with])
        coverage_rate = np.mean([r['is_covered'] for r in results_with])
    else:
        avg_set_size = 0
        coverage_rate = 0
    
    improvement = avg_acc_with - avg_acc_without
    
    logger.info(f"\n测试样本数: {len(results_without)}")
    logger.info(f"\n无Conformal ASR:")
    logger.info(f"  平均准确率: {avg_acc_without:.2%}")
    logger.info(f"  平均WER:   {avg_wer_without:.2%}")
    logger.info(f"\n使用Conformal ASR:")
    logger.info(f"  平均准确率: {avg_acc_with:.2%}")
    logger.info(f"  平均WER:   {avg_wer_with:.2%}")
    logger.info(f"  平均预测集大小: {avg_set_size:.2f}")
    logger.info(f"  覆盖率: {coverage_rate:.2%}")
    logger.info(f"\n改善:")
    logger.info(f"  准确率提升: {improvement:+.2%}")
    logger.info(f"  WER降低: {avg_wer_without - avg_wer_with:+.2%}")
    
    # 保存结果
    results = {
        'test_size': len(results_without),
        'calibration_size': len(pairs) - len(results_without),
        'results_without_conformal': results_without,
        'results_with_conformal': results_with,
        'statistics': {
            'avg_accuracy_without': float(avg_acc_without),
            'avg_accuracy_with': float(avg_acc_with),
            'avg_wer_without': float(avg_wer_without),
            'avg_wer_with': float(avg_wer_with),
            'avg_set_size': float(avg_set_size),
            'coverage_rate': float(coverage_rate),
            'improvement': float(improvement)
        }
    }
    
    results_path = output_dir / 'elderly_video_asr_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n✅ 结果已保存到: {results_path}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()

