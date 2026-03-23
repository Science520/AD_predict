#!/usr/bin/env python3
"""
分析实验结果并生成对比报告
"""
import os
import json
import argparse
from pathlib import Path
import pandas as pd


def parse_log_file(log_path):
    """从日志文件中解析训练结果"""
    results = {
        'wer': None,
        'train_loss': None,
        'eval_loss': None,
        'train_time': None,
        'epochs': None
    }
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # 提取WER
        import re
        wer_matches = re.findall(r"'eval_wer':\s*([0-9.]+)", content)
        if wer_matches:
            results['wer'] = float(wer_matches[-1])
        
        # 提取训练loss
        loss_matches = re.findall(r"'train_loss':\s*([0-9.]+)", content)
        if loss_matches:
            results['train_loss'] = float(loss_matches[-1])
        
        # 提取eval loss
        eval_loss_matches = re.findall(r"'eval_loss':\s*([0-9.]+)", content)
        if eval_loss_matches:
            results['eval_loss'] = float(eval_loss_matches[-1])
        
        # 提取训练时间
        time_matches = re.findall(r"'train_runtime':\s*([0-9.]+)", content)
        if time_matches:
            results['train_time'] = float(time_matches[-1]) / 60  # 转换为分钟
        
        # 提取epoch数
        epoch_matches = re.findall(r"'epoch':\s*([0-9.]+)", content)
        if epoch_matches:
            results['epochs'] = float(epoch_matches[-1])
    
    return results


def load_experiment_results(results_dir):
    """加载所有实验结果"""
    results_path = Path(results_dir)
    
    experiments = []
    
    # Baseline结果
    baseline_log = Path("/tmp/whisper_logs/whisper_medium_20251022_093636.log")
    if baseline_log.exists():
        baseline_results = parse_log_file(baseline_log)
        experiments.append({
            'name': 'baseline',
            'description': 'Baseline (r=8, lr=3e-5)',
            **baseline_results
        })
    
    # 其他实验结果
    for log_file in results_path.glob("*.log"):
        exp_name = log_file.stem
        results = parse_log_file(log_file)
        
        desc_map = {
            'exp1_high_rank': '高LoRA rank (r=32, lr=5e-5)',
            'exp2_low_lr': '低学习率 (r=16, lr=1e-5)',
            'exp3_large_batch': '大batch (batch=2×8)',
            'exp4_aggressive': '激进训练 (r=64, lr=1e-4)'
        }
        
        experiments.append({
            'name': exp_name,
            'description': desc_map.get(exp_name, exp_name),
            **results
        })
    
    return experiments


def generate_report(experiments, output_path):
    """生成实验对比报告"""
    df = pd.DataFrame(experiments)
    
    report = []
    report.append("=" * 80)
    report.append("Whisper方言ASR参数调优实验报告")
    report.append("=" * 80)
    report.append("")
    
    # 数据表格
    report.append("## 实验结果对比")
    report.append("")
    report.append(df.to_string(index=False))
    report.append("")
    
    # 找出最佳配置
    if 'wer' in df.columns and df['wer'].notna().any():
        best_idx = df['wer'].idxmin()
        best_exp = df.iloc[best_idx]
        
        report.append("## 🏆 最佳配置")
        report.append("")
        report.append(f"实验名称: {best_exp['name']}")
        report.append(f"描述: {best_exp['description']}")
        report.append(f"WER: {best_exp['wer']:.4f}")
        report.append(f"训练Loss: {best_exp['train_loss']:.4f}")
        report.append(f"训练时间: {best_exp['train_time']:.1f}分钟")
        report.append("")
        
        # 与baseline对比
        if 'baseline' in df['name'].values:
            baseline = df[df['name'] == 'baseline'].iloc[0]
            improvement = (baseline['wer'] - best_exp['wer']) / baseline['wer'] * 100
            
            report.append("## 📊 改进幅度")
            report.append("")
            report.append(f"Baseline WER: {baseline['wer']:.4f}")
            report.append(f"最佳 WER: {best_exp['wer']:.4f}")
            report.append(f"改进: {improvement:.2f}%")
            report.append("")
    
    # 保存报告
    report_text = "\n".join(report)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n报告已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="分析实验结果")
    parser.add_argument('--results_dir', type=str, required=True,
                      help="实验结果目录")
    args = parser.parse_args()
    
    experiments = load_experiment_results(args.results_dir)
    output_path = os.path.join(args.results_dir, "EXPERIMENT_REPORT.md")
    generate_report(experiments, output_path)


if __name__ == "__main__":
    main()

