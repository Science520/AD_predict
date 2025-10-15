#!/usr/bin/env python3
"""
带重排序的Conformal ASR评估

实现思路：
1. CI生成候选集
2. 使用可训练的重排序模型选择最佳候选
3. 融合方言特征、语言模型分数等
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import logging
import whisper
from glob import glob
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_wer(reference, hypothesis):
    """计算词错误率"""
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


class CandidateReranker(nn.Module):
    """
    可训练的候选重排序模型
    
    输入特征：
    - ASR置信度
    - 候选长度
    - 特殊字符比例
    - 方言特征匹配度
    - 语言模型困惑度（如果有）
    """
    
    def __init__(self, feature_dim=8):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # 简单的MLP排序器
        self.ranker = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # 输出排序分数
        )
    
    def extract_features(self, candidate: str, asr_logprob: float = None) -> torch.Tensor:
        """
        从候选文本提取特征
        
        特征包括：
        1. 文本长度（归一化）
        2. 特殊字符比例
        3. 数字比例
        4. 标点符号比例
        5. ASR log概率（如果有）
        6. 重复字符比例
        7. 空格比例
        8. 平均字符频率
        """
        features = []
        
        # 1. 长度特征（归一化到0-1）
        length_norm = min(len(candidate) / 200.0, 1.0)
        features.append(length_norm)
        
        # 2. 特殊字符比例
        special_chars = sum(1 for c in candidate if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / max(len(candidate), 1)
        features.append(special_ratio)
        
        # 3. 数字比例
        digit_ratio = sum(1 for c in candidate if c.isdigit()) / max(len(candidate), 1)
        features.append(digit_ratio)
        
        # 4. 标点符号比例
        punctuation = "，。！？、；：""''（）【】《》"
        punct_ratio = sum(1 for c in candidate if c in punctuation) / max(len(candidate), 1)
        features.append(punct_ratio)
        
        # 5. ASR log概率（如果有，否则用0）
        asr_score = asr_logprob if asr_logprob is not None else 0.0
        features.append(asr_score)
        
        # 6. 重复字符比例
        unique_chars = len(set(candidate))
        repeat_ratio = 1.0 - (unique_chars / max(len(candidate), 1))
        features.append(repeat_ratio)
        
        # 7. 空格比例
        space_ratio = candidate.count(' ') / max(len(candidate), 1)
        features.append(space_ratio)
        
        # 8. 平均字符频率（简化：用长度倒数）
        avg_freq = 1.0 / max(len(candidate), 1)
        features.append(avg_freq)
        
        return torch.FloatTensor(features)
    
    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        """
        输入: (batch_size, feature_dim)
        输出: (batch_size, 1) 排序分数
        """
        return self.ranker(candidate_features)
    
    def rank_candidates(self, candidates: List[str], asr_logprobs: List[float] = None) -> List[Tuple[str, float]]:
        """
        对候选进行排序
        
        返回: [(候选文本, 排序分数), ...] 按分数从高到低排序
        """
        if asr_logprobs is None:
            asr_logprobs = [None] * len(candidates)
        
        # 提取特征
        features = []
        for cand, logprob in zip(candidates, asr_logprobs):
            feat = self.extract_features(cand, logprob)
            features.append(feat)
        
        features = torch.stack(features)  # (n_candidates, feature_dim)
        
        # 计算排序分数
        with torch.no_grad():
            scores = self.forward(features).squeeze(-1)  # (n_candidates,)
        
        # 排序
        ranked = sorted(zip(candidates, scores.tolist()), key=lambda x: x[1], reverse=True)
        
        return ranked


class ConformalASRWithReranking:
    """带重排序的Conformal ASR"""
    
    def __init__(self, model_name='large-v3', coverage=0.95):
        logger.info(f"加载Whisper模型: {model_name}...")
        self.model = whisper.load_model(model_name)
        self.coverage = coverage
        self.calibrated_threshold = None
        
        # 创建重排序模型
        self.reranker = CandidateReranker(feature_dim=8)
        logger.info("✅ 模型加载完成")
    
    def transcribe_with_candidates(self, audio_path, n_candidates=5):
        """
        生成多个候选转录
        
        策略：
        1. 使用不同temperature
        2. 使用beam search的多个beam
        """
        candidates = []
        logprobs = []
        
        # Temperature sampling
        for temp in [0.0, 0.2, 0.4, 0.6, 0.8]:
            try:
                result = self.model.transcribe(
                    audio_path,
                    language='zh',
                    task='transcribe',
                    temperature=temp,
                    verbose=False
                )
                text = result['text']
                
                if text not in candidates:
                    candidates.append(text)
                    # 简化：用负的temperature作为logprob的代理
                    logprobs.append(-temp)
                
                if len(candidates) >= n_candidates:
                    break
            except:
                pass
        
        return candidates, logprobs
    
    def calibrate(self, calibration_pairs):
        """校准Conformal模型"""
        logger.info(f"\n校准Conformal模型（{len(calibration_pairs)}个样本）...")
        
        scores = []
        for pair in calibration_pairs:
            audio_path = pair['audio_path']
            ground_truth = pair['subtitle_text']
            
            candidates, _ = self.transcribe_with_candidates(audio_path, n_candidates=5)
            
            # 计算非一致性分数
            best_wer = min(compute_wer(ground_truth, cand) for cand in candidates)
            scores.append(best_wer)
        
        scores = np.array(scores)
        n = len(scores)
        q = min((n + 1) * self.coverage / n, 1.0)
        self.calibrated_threshold = np.quantile(scores, q)
        
        logger.info(f"✅ 校准完成，阈值: {self.calibrated_threshold:.4f}")
    
    def train_reranker(self, training_pairs):
        """
        训练重排序模型
        
        监督信号：选择WER最低的候选
        """
        logger.info(f"\n训练重排序模型（{len(training_pairs)}个样本）...")
        
        optimizer = torch.optim.Adam(self.reranker.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        
        n_epochs = 10
        
        for epoch in range(n_epochs):
            total_loss = 0.0
            n_batches = 0
            
            for pair in training_pairs:
                audio_path = pair['audio_path']
                ground_truth = pair['subtitle_text']
                
                # 生成候选
                candidates, logprobs = self.transcribe_with_candidates(audio_path, n_candidates=5)
                
                if len(candidates) < 2:
                    continue
                
                # 计算每个候选的WER
                wers = [compute_wer(ground_truth, cand) for cand in candidates]
                
                # 找到最佳候选的索引
                best_idx = np.argmin(wers)
                
                # 创建标签（最佳候选=1，其他=0）
                labels = torch.zeros(len(candidates))
                labels[best_idx] = 1.0
                
                # 提取特征
                features = []
                for cand, logprob in zip(candidates, logprobs):
                    feat = self.reranker.extract_features(cand, logprob)
                    features.append(feat)
                
                features = torch.stack(features)
                
                # 前向传播
                scores = self.reranker(features).squeeze(-1)
                
                # 计算损失（二分类）
                loss = criterion(scores, labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / max(n_batches, 1)
            logger.info(f"  Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
        
        logger.info("✅ 重排序模型训练完成")
    
    def predict(self, audio_path, use_reranking=False):
        """
        预测（带可选的重排序）
        """
        # 生成候选集
        candidates, logprobs = self.transcribe_with_candidates(audio_path, n_candidates=5)
        
        if use_reranking:
            # 使用重排序模型选择最佳候选
            ranked = self.reranker.rank_candidates(candidates, logprobs)
            best_candidate = ranked[0][0]  # 分数最高的
            rerank_score = ranked[0][1]
        else:
            # 默认选择第一个（temperature=0）
            best_candidate = candidates[0]
            rerank_score = None
        
        return {
            'prediction': best_candidate,
            'candidates': candidates,
            'n_candidates': len(candidates),
            'rerank_score': rerank_score
        }


def main():
    logger.info("="*60)
    logger.info("带重排序的Conformal ASR评估")
    logger.info("="*60)
    
    # 配置
    audio_dir = "data/raw/audio/elderly_audios"
    subtitle_dir = "data/raw/audio/result"
    output_dir = Path("experiments/reranking_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载音频-字幕配对
    logger.info("\n加载数据...")
    manifest_path = Path(audio_dir) / 'audio_manifest.json'
    with open(manifest_path, 'r', encoding='utf-8') as f:
        audio_manifest = json.load(f)
    
    pairs = []
    for item in audio_manifest:
        idx = item['index']
        audio_path = item['audio_path']
        subtitle_path = Path(subtitle_dir) / f"test{idx}.txt"
        
        if subtitle_path.exists():
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                subtitle_text = f.read().strip()
            pairs.append({
                'index': idx,
                'audio_path': audio_path,
                'subtitle_text': subtitle_text
            })
    
    logger.info(f"加载了 {len(pairs)} 对音频-字幕")
    
    if len(pairs) < 4:
        logger.error("数据不足（至少需要4对）")
        return
    
    # 分割数据
    n_train = int(len(pairs) * 0.4)
    n_calib = int(len(pairs) * 0.2)
    
    train_pairs = pairs[:n_train]
    calib_pairs = pairs[n_train:n_train+n_calib]
    test_pairs = pairs[n_train+n_calib:]
    
    logger.info(f"\n数据分割:")
    logger.info(f"  训练集（重排序）: {len(train_pairs)}")
    logger.info(f"  校准集（CI）: {len(calib_pairs)}")
    logger.info(f"  测试集: {len(test_pairs)}")
    
    # 创建模型
    model = ConformalASRWithReranking(model_name='large-v3', coverage=0.95)
    
    # 校准CI
    if len(calib_pairs) > 0:
        model.calibrate(calib_pairs)
    
    # 训练重排序模型
    if len(train_pairs) > 0:
        model.train_reranker(train_pairs)
    
    # 评估测试集
    logger.info(f"\n{'='*60}")
    logger.info("评估测试集")
    logger.info(f"{'='*60}")
    
    results = []
    
    for i, pair in enumerate(test_pairs, 1):
        logger.info(f"\n[{i}/{len(test_pairs)}] {Path(pair['audio_path']).name}")
        
        # 无重排序
        result_no_rerank = model.predict(pair['audio_path'], use_reranking=False)
        wer_no_rerank = compute_wer(pair['subtitle_text'], result_no_rerank['prediction'])
        
        # 有重排序
        result_rerank = model.predict(pair['audio_path'], use_reranking=True)
        wer_rerank = compute_wer(pair['subtitle_text'], result_rerank['prediction'])
        
        logger.info(f"  真实字幕: {pair['subtitle_text'][:60]}...")
        logger.info(f"  无重排序: {result_no_rerank['prediction'][:60]}...")
        logger.info(f"    WER: {wer_no_rerank:.2%}")
        logger.info(f"  有重排序: {result_rerank['prediction'][:60]}...")
        logger.info(f"    WER: {wer_rerank:.2%}")
        logger.info(f"  改善: {(wer_no_rerank - wer_rerank)*100:+.2f}%")
        
        results.append({
            'index': pair['index'],
            'ground_truth': pair['subtitle_text'],
            'pred_no_rerank': result_no_rerank['prediction'],
            'pred_rerank': result_rerank['prediction'],
            'wer_no_rerank': wer_no_rerank,
            'wer_rerank': wer_rerank,
            'improvement': wer_no_rerank - wer_rerank,
            'n_candidates': result_rerank['n_candidates']
        })
    
    # 统计
    logger.info(f"\n{'='*60}")
    logger.info("结果统计")
    logger.info(f"{'='*60}")
    
    avg_wer_no_rerank = np.mean([r['wer_no_rerank'] for r in results])
    avg_wer_rerank = np.mean([r['wer_rerank'] for r in results])
    avg_improvement = np.mean([r['improvement'] for r in results])
    
    improved_count = sum(1 for r in results if r['improvement'] > 0)
    
    logger.info(f"\n测试样本数: {len(results)}")
    logger.info(f"\n无重排序:")
    logger.info(f"  平均WER: {avg_wer_no_rerank:.2%}")
    logger.info(f"  平均准确率: {1-avg_wer_no_rerank:.2%}")
    logger.info(f"\n有重排序:")
    logger.info(f"  平均WER: {avg_wer_rerank:.2%}")
    logger.info(f"  平均准确率: {1-avg_wer_rerank:.2%}")
    logger.info(f"\n改善:")
    logger.info(f"  WER平均降低: {avg_improvement*100:.2f}%")
    logger.info(f"  准确率平均提升: {avg_improvement*100:.2f}%")
    logger.info(f"  改善样本数: {improved_count}/{len(results)} ({improved_count/len(results)*100:.1f}%)")
    
    # 保存结果
    output = {
        'test_size': len(results),
        'train_size': len(train_pairs),
        'calib_size': len(calib_pairs),
        'results': results,
        'statistics': {
            'avg_wer_no_rerank': float(avg_wer_no_rerank),
            'avg_wer_rerank': float(avg_wer_rerank),
            'avg_improvement': float(avg_improvement),
            'improved_ratio': float(improved_count / len(results))
        }
    }
    
    output_path = output_dir / 'reranking_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n✅ 结果已保存到: {output_path}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()

