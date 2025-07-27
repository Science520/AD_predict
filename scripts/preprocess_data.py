#!/usr/bin/env python3
"""
数据预处理脚本
Data preprocessing script for Alzheimer detection system
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import multiprocessing as mp
from tqdm import tqdm
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.audio_processor import AudioProcessor
from src.data.eeg_processor import EEGProcessor
from src.data.text_processor import TextProcessor

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.audio_processor = AudioProcessor(config.get('data', {}))
        self.eeg_processor = EEGProcessor(config.get('data', {}))
        self.text_processor = TextProcessor(config.get('data', {}))
        
        # 输出目录
        self.output_dir = Path(config['data']['processed_data_path'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def preprocess_audio_files(self, audio_file_list: List[str], output_subdir: str):
        """批量处理音频文件"""
        output_path = self.output_dir / output_subdir / "audio_features"
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"开始处理 {len(audio_file_list)} 个音频文件...")
        
        def process_single_audio(args):
            audio_path, idx = args
            try:
                # 处理音频文件
                audio_features, speech_features = self.audio_processor.process_audio_file(audio_path)
                
                # 保存特征
                output_file = output_path / f"audio_{idx:06d}.npz"
                np.savez_compressed(
                    output_file,
                    audio_features=audio_features,
                    speech_features=speech_features,
                    original_path=str(audio_path)
                )
                
                return {
                    'index': idx,
                    'original_path': str(audio_path),
                    'feature_path': str(output_file),
                    'status': 'success'
                }
                
            except Exception as e:
                logger.error(f"处理音频文件 {audio_path} 时出错: {e}")
                return {
                    'index': idx,
                    'original_path': str(audio_path),
                    'error': str(e),
                    'status': 'failed'
                }
        
        # 并行处理
        with mp.Pool(processes=self.config.get('preprocessing', {}).get('num_workers', 4)) as pool:
            args_list = [(path, i) for i, path in enumerate(audio_file_list)]
            results = list(tqdm(
                pool.imap(process_single_audio, args_list),
                total=len(args_list),
                desc="Processing audio files"
            ))
        
        # 保存处理结果
        results_file = output_path / "processing_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        success_count = sum(1 for r in results if r['status'] == 'success')
        logger.info(f"音频处理完成: {success_count}/{len(results)} 成功")
        
        return results
    
    def preprocess_eeg_files(self, eeg_file_list: List[str], output_subdir: str):
        """批量处理EEG文件"""
        output_path = self.output_dir / output_subdir / "eeg_features"
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"开始处理 {len(eeg_file_list)} 个EEG文件...")
        
        def process_single_eeg(args):
            eeg_path, idx = args
            try:
                # 处理EEG文件
                eeg_features = self.eeg_processor.process_eeg_file(eeg_path)
                
                # 保存特征
                output_file = output_path / f"eeg_{idx:06d}.npz"
                np.savez_compressed(
                    output_file,
                    eeg_features=eeg_features,
                    original_path=str(eeg_path)
                )
                
                return {
                    'index': idx,
                    'original_path': str(eeg_path),
                    'feature_path': str(output_file),
                    'status': 'success'
                }
                
            except Exception as e:
                logger.error(f"处理EEG文件 {eeg_path} 时出错: {e}")
                return {
                    'index': idx,
                    'original_path': str(eeg_path),
                    'error': str(e),
                    'status': 'failed'
                }
        
        # 并行处理
        with mp.Pool(processes=self.config.get('preprocessing', {}).get('num_workers', 4)) as pool:
            args_list = [(path, i) for i, path in enumerate(eeg_file_list)]
            results = list(tqdm(
                pool.imap(process_single_eeg, args_list),
                total=len(args_list),
                desc="Processing EEG files"
            ))
        
        # 保存处理结果
        results_file = output_path / "processing_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        success_count = sum(1 for r in results if r['status'] == 'success')
        logger.info(f"EEG处理完成: {success_count}/{len(results)} 成功")
        
        return results
    
    def preprocess_text_data(self, text_data: List[str], output_subdir: str):
        """批量处理文本数据"""
        output_path = self.output_dir / output_subdir / "text_features"
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"开始处理 {len(text_data)} 条文本...")
        
        all_features = []
        success_count = 0
        
        for idx, text in enumerate(tqdm(text_data, desc="Processing text")):
            try:
                # 提取文本特征
                text_features = self.text_processor.extract_features(text)
                
                # 保存特征
                output_file = output_path / f"text_{idx:06d}.npz"
                np.savez_compressed(
                    output_file,
                    text_features=text_features,
                    original_text=text
                )
                
                all_features.append({
                    'index': idx,
                    'feature_path': str(output_file),
                    'text_length': len(text),
                    'status': 'success'
                })
                success_count += 1
                
            except Exception as e:
                logger.error(f"处理文本 {idx} 时出错: {e}")
                all_features.append({
                    'index': idx,
                    'error': str(e),
                    'status': 'failed'
                })
        
        # 保存处理结果
        results_file = output_path / "processing_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_features, f, ensure_ascii=False, indent=2)
        
        logger.info(f"文本处理完成: {success_count}/{len(text_data)} 成功")
        
        return all_features
    
    def create_dataset_manifest(self, data_splits: Dict[str, Dict], output_file: str):
        """创建数据集清单文件"""
        
        manifest = {
            'dataset_info': {
                'name': 'Alzheimer Detection Dataset',
                'version': '1.0',
                'description': '多模态阿尔茨海默症检测数据集',
                'modalities': ['audio', 'eeg', 'text'],
                'classes': ['Healthy', 'MCI', 'AD']
            },
            'splits': data_splits,
            'preprocessing_config': self.config
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据集清单已保存到: {output_file}")


def load_data_from_csv(csv_path: str) -> pd.DataFrame:
    """从CSV文件加载数据信息"""
    return pd.read_csv(csv_path)


def split_data(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """划分训练、验证和测试集"""
    
    # 按类别分层采样
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    for label in df['diagnosis'].unique():
        label_df = df[df['diagnosis'] == label]
        n_samples = len(label_df)
        
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        # 随机打乱
        label_df = label_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_dfs.append(label_df[:n_train])
        val_dfs.append(label_df[n_train:n_train+n_val])
        test_dfs.append(label_df[n_train+n_val:])
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description="阿尔茨海默症检测数据预处理")
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--data_csv', type=str, required=True, help='数据清单CSV文件')
    parser.add_argument('--output_dir', type=str, help='输出目录（覆盖配置文件）')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--skip_audio', action='store_true', help='跳过音频处理')
    parser.add_argument('--skip_eeg', action='store_true', help='跳过EEG处理')
    parser.add_argument('--skip_text', action='store_true', help='跳过文本处理')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 覆盖输出目录（如果指定）
    if args.output_dir:
        config['data']['processed_data_path'] = args.output_dir
    
    # 加载数据清单
    logger.info(f"加载数据清单: {args.data_csv}")
    df = load_data_from_csv(args.data_csv)
    logger.info(f"总样本数: {len(df)}")
    logger.info(f"类别分布:\n{df['diagnosis'].value_counts()}")
    
    # 划分数据集
    train_df, val_df, test_df = split_data(df)
    logger.info(f"数据划分: 训练集 {len(train_df)}, 验证集 {len(val_df)}, 测试集 {len(test_df)}")
    
    # 创建预处理器
    preprocessor = DataPreprocessor(config)
    
    data_splits = {}
    
    # 处理各个数据集分割
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        logger.info(f"处理 {split_name} 集...")
        
        split_info = {
            'size': len(split_df),
            'class_distribution': split_df['diagnosis'].value_counts().to_dict()
        }
        
        # 处理音频文件
        if not args.skip_audio and 'audio_path' in split_df.columns:
            audio_files = split_df['audio_path'].dropna().tolist()
            if audio_files:
                audio_results = preprocessor.preprocess_audio_files(audio_files, split_name)
                split_info['audio_files'] = len(audio_files)
                split_info['audio_success'] = sum(1 for r in audio_results if r['status'] == 'success')
        
        # 处理EEG文件
        if not args.skip_eeg and 'eeg_path' in split_df.columns:
            eeg_files = split_df['eeg_path'].dropna().tolist()
            if eeg_files:
                eeg_results = preprocessor.preprocess_eeg_files(eeg_files, split_name)
                split_info['eeg_files'] = len(eeg_files)
                split_info['eeg_success'] = sum(1 for r in eeg_results if r['status'] == 'success')
        
        # 处理文本数据
        if not args.skip_text and 'text' in split_df.columns:
            text_data = split_df['text'].dropna().tolist()
            if text_data:
                text_results = preprocessor.preprocess_text_data(text_data, split_name)
                split_info['text_samples'] = len(text_data)
                split_info['text_success'] = sum(1 for r in text_results if r['status'] == 'success')
        
        # 保存分割后的CSV
        split_csv_path = preprocessor.output_dir / f"{split_name}_manifest.csv"
        split_df.to_csv(split_csv_path, index=False)
        split_info['csv_path'] = str(split_csv_path)
        
        data_splits[split_name] = split_info
    
    # 创建数据集清单
    manifest_path = preprocessor.output_dir / "dataset_manifest.json"
    preprocessor.create_dataset_manifest(data_splits, manifest_path)
    
    logger.info("数据预处理完成!")
    logger.info(f"处理结果已保存到: {preprocessor.output_dir}")


if __name__ == "__main__":
    main() 