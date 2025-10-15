#!/usr/bin/env python3
"""
可视化Conformal ASR对比结果

生成详细的对比图表
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# 设置样式
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_evaluation_results(results_dir: str):
    """加载评估结果"""
    
    results_dir = Path(results_dir)
    
    # 加载详细结果
    detailed_path = results_dir / "conformal_asr_detailed_results.json"
    stats_path = results_dir / "conformal_asr_statistics.json"
    
    if not detailed_path.exists():
        logger.error(f"结果文件不存在: {detailed_path}")
        return None, None
    
    with open(detailed_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    with open(stats_path, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    
    logger.info(f"加载了 {len(results)} 条评估结果")
    
    return results, stats


def plot_accuracy_comparison(results: list, stats: dict, output_dir: Path):
    """绘制准确率对比图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 准确率对比柱状图
    ax = axes[0, 0]
    
    methods = ['无Conformal\nInference', '使用Conformal\nInference']
    accuracies = [
        stats['avg_accuracy_without'] * 100,
        stats['avg_accuracy_with'] * 100
    ]
    errors = [
        stats['std_accuracy_without'] * 100,
        stats['std_accuracy_with'] * 100
    ]
    
    bars = ax.bar(methods, accuracies, yerr=errors, 
                  color=['#FF6B6B', '#4ECDC4'],
                  alpha=0.8, edgecolor='black', linewidth=1.5,
                  capsize=10)
    
    ax.set_ylabel('平均准确率 (%)', fontsize=12, fontweight='bold')
    ax.set_title('ASR识别准确率对比', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (bar, acc, err) in enumerate(zip(bars, accuracies, errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.1f}%\n±{err:.1f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. WER对比
    ax = axes[0, 1]
    
    wers = [
        stats['avg_wer_without'] * 100,
        stats['avg_wer_with'] * 100
    ]
    
    bars = ax.bar(methods, wers,
                  color=['#FF6B6B', '#4ECDC4'],
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('平均词错误率 WER (%)', fontsize=12, fontweight='bold')
    ax.set_title('词错误率(WER)对比 - 越低越好', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, wer in zip(bars, wers):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{wer:.1f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 3. 样本级别准确率对比散点图
    ax = axes[1, 0]
    
    acc_without = [r['accuracy_without'] * 100 for r in results]
    acc_with = [r['accuracy_with'] * 100 for r in results]
    
    ax.scatter(acc_without, acc_with, alpha=0.6, s=80, 
              c=range(len(results)), cmap='viridis', edgecolors='black', linewidth=0.5)
    
    # 添加y=x参考线
    max_val = max(max(acc_without), max(acc_with))
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.7, label='y=x (无改变)')
    
    ax.set_xlabel('无Conformal准确率 (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('使用Conformal准确率 (%)', fontsize=12, fontweight='bold')
    ax.set_title('样本级别准确率对比', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 标注改善的样本数量
    improved = sum(1 for r in results if r['improvement'] > 0)
    ax.text(0.05, 0.95, f'改善样本: {improved}/{len(results)} ({improved/len(results)*100:.1f}%)',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. 准确率改善分布直方图
    ax = axes[1, 1]
    
    improvements = [r['improvement'] * 100 for r in results]
    
    n, bins, patches = ax.hist(improvements, bins=30, alpha=0.7, 
                                color='steelblue', edgecolor='black', linewidth=1)
    
    # 给正向改善的柱子上色
    for i, patch in enumerate(patches):
        if bins[i] > 0:
            patch.set_facecolor('#4ECDC4')
        elif bins[i] < 0:
            patch.set_facecolor('#FF6B6B')
    
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='无改变')
    ax.axvline(x=np.mean(improvements), color='green', linestyle='-', linewidth=2, 
              alpha=0.7, label=f'平均改善: {np.mean(improvements):.2f}%')
    
    ax.set_xlabel('准确率改善 (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('样本数量', fontsize=12, fontweight='bold')
    ax.set_title('准确率改善分布', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / 'asr_accuracy_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"准确率对比图已保存: {output_path}")
    plt.close()


def plot_conformal_metrics(results: list, stats: dict, output_dir: Path):
    """绘制Conformal特定指标"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 预测集大小分布
    ax = axes[0, 0]
    
    set_sizes = [r['set_size'] for r in results]
    
    ax.hist(set_sizes, bins=range(1, max(set_sizes)+2), alpha=0.7,
           color='coral', edgecolor='black', linewidth=1)
    ax.axvline(x=stats['avg_set_size'], color='red', linestyle='--', 
              linewidth=2, label=f'平均: {stats["avg_set_size"]:.2f}')
    
    ax.set_xlabel('预测集大小', fontsize=12, fontweight='bold')
    ax.set_ylabel('样本数量', fontsize=12, fontweight='bold')
    ax.set_title('Conformal预测集大小分布', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. 预测集大小 vs 准确率
    ax = axes[0, 1]
    
    acc_with = [r['accuracy_with'] * 100 for r in results]
    
    scatter = ax.scatter(set_sizes, acc_with, alpha=0.6, s=80,
                        c=acc_with, cmap='RdYlGn', vmin=0, vmax=100,
                        edgecolors='black', linewidth=0.5)
    
    # 添加趋势线
    z = np.polyfit(set_sizes, acc_with, 1)
    p = np.poly1d(z)
    ax.plot(sorted(set_sizes), p(sorted(set_sizes)), "r--", alpha=0.8, linewidth=2)
    
    ax.set_xlabel('预测集大小', fontsize=12, fontweight='bold')
    ax.set_ylabel('准确率 (%)', fontsize=12, fontweight='bold')
    ax.set_title('预测集大小与准确率关系', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('准确率 (%)', fontsize=10)
    
    # 3. 覆盖率统计
    ax = axes[1, 0]
    
    coverage_rate = stats['coverage_rate'] * 100
    not_covered_rate = (1 - stats['coverage_rate']) * 100
    
    wedges, texts, autotexts = ax.pie(
        [coverage_rate, not_covered_rate],
        labels=['真值被覆盖', '真值未覆盖'],
        autopct='%1.1f%%',
        colors=['#4ECDC4', '#FF6B6B'],
        startangle=90,
        explode=(0.05, 0),
        shadow=True
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    ax.set_title(f'Conformal预测集覆盖率\n(目标: 95%)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # 4. Conformal置信度分布
    ax = axes[1, 1]
    
    confidences = [r['conformal_confidence'] for r in results]
    
    ax.hist(confidences, bins=30, alpha=0.7, color='plum', 
           edgecolor='black', linewidth=1)
    ax.axvline(x=np.mean(confidences), color='red', linestyle='--',
              linewidth=2, label=f'平均: {np.mean(confidences):.3f}')
    
    ax.set_xlabel('Conformal置信度', fontsize=12, fontweight='bold')
    ax.set_ylabel('样本数量', fontsize=12, fontweight='bold')
    ax.set_title('Conformal置信度分布', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = output_dir / 'conformal_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Conformal指标图已保存: {output_path}")
    plt.close()


def plot_improvement_analysis(results: list, stats: dict, output_dir: Path):
    """绘制改善分析图"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Top改善和Top退化样本
    ax = axes[0]
    
    # 按改善排序
    sorted_results = sorted(results, key=lambda x: x['improvement'], reverse=True)
    
    # Top 10改善
    top_improved = sorted_results[:10]
    # Top 10退化
    top_degraded = sorted_results[-10:][::-1]
    
    # 绘制改善
    improvements = [r['improvement'] * 100 for r in top_improved]
    indices = range(len(improvements))
    
    bars1 = ax.barh(indices, improvements, color='#4ECDC4', 
                    alpha=0.8, edgecolor='black', linewidth=1,
                    label='Top 10 改善样本')
    
    # 绘制退化
    degradations = [r['improvement'] * 100 for r in top_degraded]
    indices_deg = range(len(improvements), len(improvements) + len(degradations))
    
    bars2 = ax.barh(indices_deg, degradations, color='#FF6B6B',
                    alpha=0.8, edgecolor='black', linewidth=1,
                    label='Top 10 退化样本')
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax.set_xlabel('准确率改善 (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('样本', fontsize=12, fontweight='bold')
    ax.set_title('Top改善/退化样本分析', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 2. 改善率统计
    ax = axes[1]
    
    # 计算不同程度的改善比例
    improvements_all = [r['improvement'] * 100 for r in results]
    
    categories = [
        '显著退化\n(<-5%)',
        '轻微退化\n(-5%~0%)',
        '无明显变化\n(0%~1%)',
        '轻微改善\n(1%~5%)',
        '显著改善\n(>5%)'
    ]
    
    counts = [
        sum(1 for x in improvements_all if x < -5),
        sum(1 for x in improvements_all if -5 <= x < 0),
        sum(1 for x in improvements_all if 0 <= x < 1),
        sum(1 for x in improvements_all if 1 <= x < 5),
        sum(1 for x in improvements_all if x >= 5)
    ]
    
    colors_cat = ['#d62728', '#ff7f0e', '#bcbd22', '#2ca02c', '#1f77b4']
    
    bars = ax.bar(range(len(categories)), counts, color=colors_cat,
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel('样本数量', fontsize=12, fontweight='bold')
    ax.set_title('改善程度分类统计', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percentage = count / len(results) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}\n({percentage:.1f}%)',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / 'improvement_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"改善分析图已保存: {output_path}")
    plt.close()


def generate_summary_report(results: list, stats: dict, output_dir: Path):
    """生成文本总结报告"""
    
    report = []
    report.append("="*80)
    report.append("Conformal ASR评估报告")
    report.append("="*80)
    report.append("")
    
    report.append("1. 基本信息")
    report.append(f"   评估样本数: {stats['n_samples']}")
    report.append("")
    
    report.append("2. 准确率对比")
    report.append(f"   无Conformal平均准确率:  {stats['avg_accuracy_without']*100:.2f}% (±{stats['std_accuracy_without']*100:.2f}%)")
    report.append(f"   使用Conformal平均准确率: {stats['avg_accuracy_with']*100:.2f}% (±{stats['std_accuracy_with']*100:.2f}%)")
    report.append(f"   平均提升:               {stats['avg_improvement']*100:.2f}%")
    report.append(f"   中位数提升:             {stats['median_improvement']*100:.2f}%")
    report.append(f"   正向改善样本比例:        {stats['positive_improvement_rate']*100:.2f}%")
    report.append("")
    
    report.append("3. 词错误率(WER)对比")
    report.append(f"   无Conformal平均WER:  {stats['avg_wer_without']*100:.2f}%")
    report.append(f"   使用Conformal平均WER: {stats['avg_wer_with']*100:.2f}%")
    report.append(f"   WER降低:            {(stats['avg_wer_without']-stats['avg_wer_with'])*100:.2f}%")
    report.append("")
    
    report.append("4. Conformal特定指标")
    report.append(f"   平均预测集大小:     {stats['avg_set_size']:.2f}")
    report.append(f"   覆盖率:            {stats['coverage_rate']*100:.2f}% (目标: 95%)")
    report.append("")
    
    improvements = [r['improvement'] * 100 for r in results]
    report.append("5. 改善统计")
    report.append(f"   最大改善:          {max(improvements):.2f}%")
    report.append(f"   最大退化:          {min(improvements):.2f}%")
    report.append(f"   改善标准差:         {np.std(improvements):.2f}%")
    report.append("")
    
    report.append("6. 结论")
    if stats['avg_improvement'] > 0:
        report.append(f"   ✅ Conformal Inference显著提升了ASR准确率 (+{stats['avg_improvement']*100:.2f}%)")
    else:
        report.append(f"   ⚠️ Conformal Inference未能提升ASR准确率")
    
    if stats['coverage_rate'] >= 0.90:
        report.append(f"   ✅ 覆盖率达到预期目标 ({stats['coverage_rate']*100:.2f}%)")
    else:
        report.append(f"   ⚠️ 覆盖率未达到目标 ({stats['coverage_rate']*100:.2f}% < 95%)")
    
    report.append("")
    report.append("="*80)
    
    # 保存报告
    report_text = '\n'.join(report)
    report_path = output_dir / 'evaluation_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"评估报告已保存: {report_path}")
    
    # 同时打印到控制台
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description="可视化Conformal ASR对比结果")
    parser.add_argument('--results_dir', type=str,
                       default='experiments/conformal_evaluation',
                       help='评估结果目录')
    parser.add_argument('--output_dir', type=str,
                       default='experiments/conformal_evaluation/visualizations',
                       help='可视化输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("可视化Conformal ASR对比结果")
    logger.info("="*60)
    
    # 加载结果
    logger.info("\n加载评估结果...")
    results, stats = load_evaluation_results(args.results_dir)
    
    if results is None:
        logger.error("无法加载评估结果")
        return
    
    # 生成图表
    logger.info("\n生成准确率对比图...")
    plot_accuracy_comparison(results, stats, output_dir)
    
    logger.info("\n生成Conformal指标图...")
    plot_conformal_metrics(results, stats, output_dir)
    
    logger.info("\n生成改善分析图...")
    plot_improvement_analysis(results, stats, output_dir)
    
    # 生成报告
    logger.info("\n生成评估报告...")
    generate_summary_report(results, stats, output_dir)
    
    logger.info("\n" + "="*60)
    logger.info("✅ 可视化完成！")
    logger.info(f"所有图表已保存到: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()

