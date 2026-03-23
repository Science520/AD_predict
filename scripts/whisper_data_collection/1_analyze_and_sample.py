"""
智能数据采样分析脚本
功能：
1. 分析现有数据的方言分布
2. 计算每个方言类别需要的目标样本数
3. 根据Excel中的视频信息，生成下载优先级列表
4. 输出一个采样计划JSON文件
"""

import os
import sys
import pandas as pd
import json
import yaml
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_config(config_path="configs/training_args.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def analyze_current_distribution(excel_path, audio_dir=None):
    """
    分析当前已有数据的方言分布
    
    Returns:
        dict: {
            'total': int,
            'distribution': {dialect: count},
            'downloaded': [indices]
        }
    """
    print("\n" + "="*80)
    print("步骤 1: 分析现有数据分布")
    print("="*80)
    
    # 读取Excel
    df = pd.read_excel(excel_path)
    print(f"Excel中共有 {len(df)} 条视频信息")
    
    # 统计方言分布
    dialect_counts = df['dialect_label'].value_counts().to_dict()
    
    # 检查已下载的音频
    downloaded_indices = []
    if audio_dir and os.path.exists(audio_dir):
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        for audio_file in audio_files:
            # elderly_audio_0001.wav -> 1
            idx = int(audio_file.replace('elderly_audio_', '').replace('.wav', ''))
            downloaded_indices.append(idx)
        print(f"\n已下载音频数量: {len(downloaded_indices)}")
    else:
        print(f"\n警告: 音频目录不存在或未指定")
    
    # 统计已下载数据的方言分布
    downloaded_distribution = {}
    if downloaded_indices:
        for idx in downloaded_indices:
            if idx <= len(df):
                dialect = df.iloc[idx-1]['dialect_label']
                if pd.notna(dialect):
                    downloaded_distribution[dialect] = downloaded_distribution.get(dialect, 0) + 1
    
    print(f"\n当前方言分布（Excel全量）:")
    for dialect, count in sorted(dialect_counts.items(), key=lambda x: x[1], reverse=True):
        downloaded = downloaded_distribution.get(dialect, 0)
        print(f"  {dialect:25s}: {count:4d} 个视频 (已下载: {downloaded})")
    
    return {
        'total': len(df),
        'excel_distribution': dialect_counts,
        'downloaded': downloaded_indices,
        'downloaded_distribution': downloaded_distribution
    }


def calculate_target_samples(
    current_stats,
    target_total=None,
    min_samples_per_dialect=20,
    balance_strategy='weighted'
):
    """
    计算每个方言的目标样本数
    
    Args:
        current_stats: 当前统计信息
        target_total: 目标总样本数（None则根据策略自动计算）
        min_samples_per_dialect: 每个方言的最小样本数
        balance_strategy: 平衡策略
            - 'weighted': 加权平衡（保留原始分布但增强少数类）
            - 'uniform': 均匀分布
            - 'proportional': 按原始比例扩展
    
    Returns:
        dict: {dialect: target_count}
    """
    print("\n" + "="*80)
    print("步骤 2: 计算目标样本数")
    print("="*80)
    
    excel_dist = current_stats['excel_distribution']
    downloaded_dist = current_stats['downloaded_distribution']
    
    # 计算每个方言还可以下载的最大数量
    available_by_dialect = {}
    for dialect, total in excel_dist.items():
        downloaded = downloaded_dist.get(dialect, 0)
        available = total - downloaded
        available_by_dialect[dialect] = max(0, available)
    
    print(f"\n平衡策略: {balance_strategy}")
    
    if balance_strategy == 'weighted':
        # 加权策略：少数类至少要有min_samples，多数类按比例增加
        target_samples = {}
        
        # 1. 先满足所有类别的最小要求
        for dialect in excel_dist.keys():
            current = downloaded_dist.get(dialect, 0)
            target = max(min_samples_per_dialect, current)
            target_samples[dialect] = min(target, current + available_by_dialect[dialect])
        
        # 2. 对于已经超过最小要求的，按原始比例分配额外样本
        total_excel = sum(excel_dist.values())
        extra_budget = target_total - sum(target_samples.values()) if target_total else 0
        
        if extra_budget > 0:
            for dialect, excel_count in excel_dist.items():
                proportion = excel_count / total_excel
                extra = int(extra_budget * proportion)
                target_samples[dialect] = min(
                    target_samples[dialect] + extra,
                    downloaded_dist.get(dialect, 0) + available_by_dialect[dialect]
                )
    
    elif balance_strategy == 'uniform':
        # 均匀策略：每个方言目标样本数相同
        uniform_target = target_total // len(excel_dist) if target_total else min_samples_per_dialect
        
        target_samples = {}
        for dialect in excel_dist.keys():
            current = downloaded_dist.get(dialect, 0)
            target = uniform_target
            target_samples[dialect] = min(target, current + available_by_dialect[dialect])
    
    else:  # proportional
        # 按原始比例策略
        if not target_total:
            target_total = sum(excel_dist.values())
        
        total_excel = sum(excel_dist.values())
        target_samples = {}
        
        for dialect, excel_count in excel_dist.items():
            proportion = excel_count / total_excel
            target = int(target_total * proportion)
            target_samples[dialect] = min(target, 
                                         downloaded_dist.get(dialect, 0) + available_by_dialect[dialect])
    
    # 计算需要新下载的数量
    needed_samples = {}
    for dialect, target in target_samples.items():
        current = downloaded_dist.get(dialect, 0)
        needed = max(0, target - current)
        needed_samples[dialect] = needed
    
    print(f"\n目标样本分配:")
    print(f"  {'方言':25s} {'当前':>6s} {'目标':>6s} {'需要':>6s} {'可用':>6s}")
    print("  " + "-"*70)
    
    for dialect in sorted(excel_dist.keys()):
        current = downloaded_dist.get(dialect, 0)
        target = target_samples[dialect]
        needed = needed_samples[dialect]
        available = available_by_dialect[dialect]
        
        print(f"  {dialect:25s} {current:6d} {target:6d} {needed:6d} {available:6d}")
    
    print(f"\n总计:")
    print(f"  当前已下载: {sum(downloaded_dist.values())}")
    print(f"  目标总数: {sum(target_samples.values())}")
    print(f"  需要新下载: {sum(needed_samples.values())}")
    
    return {
        'target_samples': target_samples,
        'needed_samples': needed_samples,
        'available_samples': available_by_dialect
    }


def generate_download_plan(excel_path, current_stats, target_info):
    """
    生成下载计划：为每个方言选择需要下载的视频
    
    Returns:
        list: [{
            'index': int,
            'dialect': str,
            'uploader': str,
            'title': str,
            'url': str,
            'priority': int
        }]
    """
    print("\n" + "="*80)
    print("步骤 3: 生成下载计划")
    print("="*80)
    
    df = pd.read_excel(excel_path)
    downloaded = set(current_stats['downloaded'])
    needed = target_info['needed_samples']
    
    download_plan = []
    
    # 为每个方言选择视频
    for dialect, need_count in needed.items():
        if need_count == 0:
            continue
        
        print(f"\n为方言 {dialect} 选择 {need_count} 个视频...")
        
        # 找到该方言的所有未下载视频
        dialect_videos = []
        for idx, row in df.iterrows():
            video_idx = idx + 1
            if row['dialect_label'] == dialect and video_idx not in downloaded:
                dialect_videos.append({
                    'index': video_idx,
                    'dialect': dialect,
                    'uploader': row.get('up主', 'unknown'),
                    'title': row.get('视频名称', ''),
                    'url': row.get('url', ''),
                })
        
        # 如果可用视频不足
        if len(dialect_videos) < need_count:
            print(f"  ⚠️  警告: 只有 {len(dialect_videos)} 个可用视频（需要 {need_count} 个）")
            selected = dialect_videos
        else:
            # 随机选择（可以改进为更智能的选择策略，如选择时长适中的）
            import random
            random.seed(42)
            selected = random.sample(dialect_videos, need_count)
        
        # 添加到下载计划
        for video in selected:
            download_plan.append(video)
        
        print(f"  ✅ 已选择 {len(selected)} 个视频")
    
    # 按方言分组显示
    print(f"\n下载计划汇总:")
    by_dialect = defaultdict(list)
    for item in download_plan:
        by_dialect[item['dialect']].append(item)
    
    for dialect, videos in sorted(by_dialect.items()):
        print(f"  {dialect}: {len(videos)} 个视频")
        for v in videos[:3]:  # 显示前3个
            print(f"    - [{v['index']:4d}] {v['title'][:40]}")
        if len(videos) > 3:
            print(f"    ... 还有 {len(videos)-3} 个")
    
    print(f"\n总计: {len(download_plan)} 个视频需要下载")
    
    return download_plan


def save_sampling_plan(plan_data, output_path="data/sampling_plan.json"):
    """保存采样计划到JSON文件"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(plan_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n采样计划已保存到: {output_path}")
    
    # 同时保存一个简单的索引列表（方便下载脚本使用）
    indices_file = output_path.replace('.json', '_indices.txt')
    with open(indices_file, 'w') as f:
        for item in plan_data['download_plan']:
            f.write(f"{item['index']}\n")
    
    print(f"视频索引列表已保存到: {indices_file}")


def main():
    """主函数"""
    print("="*80)
    print("智能数据采样分析")
    print("="*80)
    
    # 配置
    excel_path = "/data/AD_predict/data/raw/audio/老人视频信息_final_complete_20251016_214400.xlsx"
    audio_dir = "/data/AD_predict/data/raw/audio/elderly_audios"
    
    # 可以通过配置调整这些参数
    target_total_samples = None  # None表示自动计算
    min_samples_per_dialect = 30  # 每个方言最少30个样本
    balance_strategy = 'weighted'  # 'weighted', 'uniform', 'proportional'
    
    # 1. 分析现有分布
    current_stats = analyze_current_distribution(excel_path, audio_dir)
    
    # 2. 计算目标样本数
    target_info = calculate_target_samples(
        current_stats,
        target_total=target_total_samples,
        min_samples_per_dialect=min_samples_per_dialect,
        balance_strategy=balance_strategy
    )
    
    # 3. 生成下载计划
    download_plan = generate_download_plan(excel_path, current_stats, target_info)
    
    # 4. 保存计划
    sampling_plan = {
        'config': {
            'target_total_samples': target_total_samples,
            'min_samples_per_dialect': min_samples_per_dialect,
            'balance_strategy': balance_strategy
        },
        'current_stats': {
            'total_videos': current_stats['total'],
            'downloaded_count': len(current_stats['downloaded']),
            'excel_distribution': current_stats['excel_distribution'],
            'downloaded_distribution': current_stats['downloaded_distribution']
        },
        'target_info': {
            'target_samples': target_info['target_samples'],
            'needed_samples': target_info['needed_samples']
        },
        'download_plan': download_plan
    }
    
    output_path = "data/sampling_plan.json"
    save_sampling_plan(sampling_plan, output_path)
    
    print("\n" + "="*80)
    print("✅ 采样分析完成！")
    print("="*80)
    print("\n下一步:")
    print("  1. 检查采样计划: cat data/sampling_plan.json")
    print("  2. 执行选择性下载: python scripts/whisper_data_collection/2_selective_download.py")


if __name__ == "__main__":
    main()

