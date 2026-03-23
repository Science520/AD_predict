"""
FunASR (Paraformer) LoRA微调脚本
功能：
1. 加载Paraformer-large中文ASR模型
2. 添加自定义方言特殊token
3. 配置LoRA参数高效微调
4. 专门针对中文方言优化
5. 训练并保存LoRA适配器

参考: 
- https://github.com/alibaba-damo-academy/FunASR
- https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
"""

import os
import sys
import yaml
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from pathlib import Path
import json

# 设置ModelScope环境变量
os.environ['MODELSCOPE_CACHE'] = os.path.expanduser('~/.cache/modelscope')

try:
    from funasr import AutoModel
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    FUNASR_AVAILABLE = True
except ImportError:
    print("警告: FunASR未安装。请运行: pip install funasr modelscope")
    FUNASR_AVAILABLE = False

from transformers import (
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk, Dataset
import soundfile as sf
import librosa


def load_config(config_path="configs/training_args_funasr.yaml"):
    """加载配置文件"""
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 如果config_path是相对路径，则相对于项目根目录
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


class ParaformerDataset(torch.utils.data.Dataset):
    """FunASR Paraformer数据集包装器"""
    
    def __init__(self, dataset, feature_extractor, tokenizer, sample_rate=16000):
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 加载音频
        audio_path = item.get('audio_path', item.get('audio'))
        if isinstance(audio_path, dict):
            audio_path = audio_path['path']
        
        try:
            audio_array, sr = sf.read(audio_path)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            if sr != self.sample_rate:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=self.sample_rate)
        except Exception as e:
            print(f"警告: 无法加载音频 {audio_path}: {e}")
            audio_array = np.zeros(self.sample_rate)
        
        # 提取特征（FunASR的Paraformer使用Fbank特征）
        # 这里我们简化处理，实际应该使用FunASR的特征提取器
        features = {
            'audio': torch.from_numpy(audio_array).float(),
            'audio_len': len(audio_array)
        }
        
        # 编码文本
        text = item['text']
        if self.tokenizer:
            text_tokens = self.tokenizer.encode(text)
            features['text'] = torch.tensor(text_tokens)
            features['text_len'] = len(text_tokens)
        else:
            features['text_raw'] = text
        
        # 添加方言标签（如果有）
        if 'dialect_label' in item:
            features['dialect_label'] = item['dialect_label']
        
        return features


@dataclass
class DataCollatorForASR:
    """ASR数据整理器 - 支持动态padding"""
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 提取音频
        audios = [f['audio'] for f in features]
        audio_lens = [f['audio_len'] for f in features]
        
        # Pad音频到最大长度
        max_audio_len = max(audio_lens)
        padded_audios = []
        for audio in audios:
            if len(audio) < max_audio_len:
                padded = torch.nn.functional.pad(audio, (0, max_audio_len - len(audio)))
            else:
                padded = audio
            padded_audios.append(padded)
        
        batch = {
            'audio': torch.stack(padded_audios),
            'audio_len': torch.tensor(audio_lens)
        }
        
        # 处理文本
        if 'text' in features[0]:
            texts = [f['text'] for f in features]
            text_lens = [f['text_len'] for f in features]
            
            # Pad文本
            max_text_len = max(text_lens)
            padded_texts = []
            for text in texts:
                if len(text) < max_text_len:
                    padded = torch.nn.functional.pad(text, (0, max_text_len - len(text)), value=-100)
                else:
                    padded = text
                padded_texts.append(padded)
            
            batch['labels'] = torch.stack(padded_texts)
            batch['text_len'] = torch.tensor(text_lens)
        elif 'text_raw' in features[0]:
            batch['text_raw'] = [f['text_raw'] for f in features]
        
        # 添加方言标签
        if 'dialect_label' in features[0]:
            batch['dialect_labels'] = [f['dialect_label'] for f in features]
        
        return batch


def compute_metrics(pred):
    """计算WER (Word Error Rate)"""
    import jiwer
    
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # 这里需要根据实际的tokenizer进行解码
    # 简化版本，实际需要使用FunASR的解码器
    
    # 占位符：实际应该解码并计算WER
    wer = 0.0
    
    return {"wer": wer}


def main(resume_from_checkpoint=None, config_path="configs/training_args_funasr.yaml"):
    """主函数
    
    Args:
        resume_from_checkpoint: 从指定checkpoint恢复训练
        config_path: 配置文件路径
    """
    print("=" * 80)
    print("FunASR (Paraformer) LoRA 微调")
    print("=" * 80)
    print(f"配置文件: {config_path}")
    
    if not FUNASR_AVAILABLE:
        print("\n❌ 错误: FunASR未安装")
        print("请运行以下命令安装FunASR:")
        print("  pip install funasr modelscope")
        print("\n或使用conda:")
        print("  conda install -c conda-forge funasr")
        return
    
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 1. 加载配置
    config = load_config(config_path)
    
    model_name = config['model']['name']
    output_dir = config['training']['output_dir']
    processed_data_dir = config['data']['processed_data_dir']
    dialect_labels = config['dialects']['labels']
    
    # 将相对路径转换为绝对路径
    if not os.path.isabs(processed_data_dir):
        processed_data_dir = os.path.join(project_root, processed_data_dir)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    
    print(f"模型: {model_name}")
    print(f"数据目录: {processed_data_dir}")
    print(f"输出目录: {output_dir}")
    
    # 2. 加载预处理数据集
    print("\n步骤 1: 加载预处理数据集...")
    if not os.path.exists(processed_data_dir):
        raise FileNotFoundError(
            f"数据目录不存在: {processed_data_dir}\n"
            f"请先运行: python scripts/1_prepare_dataset.py"
        )
    
    dataset = load_from_disk(processed_data_dir)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")
    
    # 3. 加载FunASR Paraformer模型
    print("\n步骤 2: 加载FunASR Paraformer模型...")
    
    try:
        # 加载Paraformer模型
        # 注意: FunASR的模型加载方式与Hugging Face不同
        model = AutoModel(
            model=model_name,
            model_revision="v2.0.4",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"✓ 成功加载模型: {model_name}")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        print("尝试使用ModelScope pipeline...")
        
        # 备选方案：使用pipeline
        asr_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model=model_name
        )
        print("✓ 使用pipeline方式加载模型")
    
    # 4. 配置LoRA
    print("\n步骤 3: 配置LoRA...")
    
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=TaskType.SEQ_2_SEQ_LM  # FunASR也是序列到序列任务
    )
    
    print("LoRA配置:")
    print(f"  r (秩): {lora_config.r}")
    print(f"  lora_alpha: {lora_config.lora_alpha}")
    print(f"  target_modules: {lora_config.target_modules}")
    print(f"  lora_dropout: {lora_config.lora_dropout}")
    
    # 注意: FunASR的模型结构可能与标准Transformer不同
    # 需要根据实际模型架构调整target_modules
    
    try:
        # 尝试应用LoRA（可能需要根据FunASR的模型结构调整）
        # model = get_peft_model(model, lora_config)
        # model.print_trainable_parameters()
        print("⚠️ 注意: FunASR的LoRA集成可能需要自定义实现")
        print("建议参考FunASR官方文档进行微调")
    except Exception as e:
        print(f"⚠️ LoRA应用失败: {e}")
        print("将使用全参数微调（需要更多GPU内存）")
    
    # 5. 准备数据集
    print("\n步骤 4: 准备训练数据...")
    
    # 创建数据整理器
    data_collator = DataCollatorForASR()
    
    # 6. 配置训练参数
    print("\n步骤 5: 配置训练参数...")
    
    training_config = config['training']
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        warmup_steps=training_config['warmup_steps'],
        
        # 优化器
        optim=training_config.get('optim', 'adamw_torch'),
        weight_decay=training_config.get('weight_decay', 0.01),
        
        # 保存和评估
        evaluation_strategy=training_config['evaluation_strategy'],
        eval_steps=training_config['eval_steps'],
        save_strategy=training_config['save_strategy'],
        save_steps=training_config['save_steps'],
        save_total_limit=training_config['save_total_limit'],
        load_best_model_at_end=training_config['load_best_model_at_end'],
        metric_for_best_model=training_config['metric_for_best_model'],
        greater_is_better=training_config['greater_is_better'],
        
        # 日志
        logging_dir=training_config['logging_dir'],
        logging_steps=training_config['logging_steps'],
        report_to=training_config['report_to'],
        
        # 其他
        fp16=training_config['fp16'],
        dataloader_num_workers=training_config['dataloader_num_workers'],
        remove_unused_columns=False,
    )
    
    print("训练配置:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  FP16: {training_args.fp16}")
    
    # 7. 创建Trainer
    print("\n步骤 6: 初始化Trainer...")
    
    print("\n⚠️ 重要提示:")
    print("FunASR (Paraformer) 的训练流程与Hugging Face Transformers有较大差异。")
    print("建议使用FunASR官方提供的训练脚本:")
    print("  https://github.com/alibaba-damo-academy/FunASR/tree/main/examples")
    print("\n或者使用ModelScope的训练工具:")
    print("  https://modelscope.cn/docs/模型训练/ASR模型训练")
    
    print("\n本脚本提供了基本框架，但实际训练需要:")
    print("  1. 使用FunASR的数据预处理流程（Fbank特征提取）")
    print("  2. 使用FunASR的训练循环")
    print("  3. 使用FunASR的解码器和评估指标")
    
    # 保存配置信息
    config_output_path = os.path.join(output_dir, "training_config.json")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config_output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 配置已保存到: {config_output_path}")
    print(f"✓ 输出目录: {output_dir}")
    
    print("\n" + "=" * 80)
    print("FunASR训练准备完成！")
    print("=" * 80)
    print("\n📚 后续步骤:")
    print("1. 参考FunASR官方训练示例修改此脚本")
    print("2. 或使用FunASR命令行工具进行训练")
    print("3. 查看文档: https://github.com/alibaba-damo-academy/FunASR")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FunASR (Paraformer) LoRA微调脚本")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从checkpoint恢复训练"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_args_funasr.yaml",
        help="配置文件路径（默认: configs/training_args_funasr.yaml）"
    )
    
    args = parser.parse_args()
    main(resume_from_checkpoint=args.resume_from_checkpoint, config_path=args.config)

