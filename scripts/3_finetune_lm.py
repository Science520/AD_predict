"""
方言感知语言模型微调脚本（可选）
功能：
1. 微调中文BERT模型以理解各地方言的用词习惯
2. 使用Masked Language Modeling (MLM)任务
3. 可用于ASR结果的后处理和重排序
"""

import os
import sys
import yaml
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
import pandas as pd
from pathlib import Path


def load_config(config_path="configs/training_args.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_transcripts_from_asr_data(processed_data_dir):
    """
    从ASR处理后的数据集中提取转录文本
    
    Args:
        processed_data_dir: ASR预处理数据目录
        
    Returns:
        list: 转录文本列表
    """
    if not os.path.exists(processed_data_dir):
        print(f"警告: ASR数据目录不存在: {processed_data_dir}")
        return []
    
    # 加载ASR数据集
    dataset = load_from_disk(processed_data_dir)
    
    transcripts = []
    
    # 从训练集和验证集中提取文本
    for split in ['train', 'validation']:
        if split in dataset:
            for item in dataset[split]:
                # 提取原始文本（去除Whisper特殊token）
                text = item['text']
                
                # 移除Whisper的特殊格式
                # 格式: <|startoftranscript|><|zh|><|transcribe|><|notimestamps|><|dialect:XXX|> TEXT
                if '|>' in text:
                    # 找到最后一个特殊token后的文本
                    parts = text.split('|>')
                    if len(parts) > 1:
                        clean_text = parts[-1].strip()
                        if clean_text:
                            transcripts.append(clean_text)
    
    print(f"从ASR数据中提取了 {len(transcripts)} 条转录文本")
    return transcripts


def load_additional_dialect_corpus(corpus_dir=None):
    """
    加载额外的方言文本语料（可选）
    
    Args:
        corpus_dir: 方言语料目录
        
    Returns:
        list: 额外的文本列表
    """
    if corpus_dir is None or not os.path.exists(corpus_dir):
        print("没有额外的方言语料")
        return []
    
    texts = []
    
    # 递归查找所有.txt文件
    for txt_file in Path(corpus_dir).rglob('*.txt'):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    # 按行分割
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    texts.extend(lines)
        except Exception as e:
            print(f"读取文件失败 {txt_file}: {e}")
            continue
    
    print(f"从语料库加载了 {len(texts)} 条额外文本")
    return texts


def prepare_mlm_dataset(texts, tokenizer, max_length=128, test_size=0.1):
    """
    准备MLM训练数据集
    
    Args:
        texts: 文本列表
        tokenizer: BERT分词器
        max_length: 最大序列长度
        test_size: 验证集比例
        
    Returns:
        DatasetDict: 包含训练集和验证集的数据集
    """
    # 创建Dataset
    dataset = Dataset.from_dict({"text": texts})
    
    # 分割训练集和验证集
    dataset = dataset.train_test_split(test_size=test_size, seed=42)
    
    # Tokenization函数
    def tokenize_function(examples):
        # 对文本进行分词，并截断/填充到max_length
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True
        )
    
    # 应用分词
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing texts"
    )
    
    return tokenized_dataset


def main():
    """主函数"""
    print("=" * 80)
    print("方言感知语言模型微调（可选）")
    print("=" * 80)
    
    # 1. 加载配置
    config = load_config()
    processed_data_dir = config['data']['processed_data_dir']
    
    # LM微调配置
    lm_model_name = "bert-base-chinese"  # 中文BERT基础模型
    lm_output_dir = "./dialect_aware_lm"
    max_seq_length = 128
    num_train_epochs = 3
    batch_size = 16
    learning_rate = 5e-5
    
    print(f"\n语言模型: {lm_model_name}")
    print(f"输出目录: {lm_output_dir}")
    
    # 2. 收集训练文本
    print("\n步骤 1: 收集训练文本...")
    
    # 从ASR数据中提取文本
    asr_texts = load_transcripts_from_asr_data(processed_data_dir)
    
    # 加载额外的方言语料（可选）
    additional_corpus_dir = None  # 用户可以指定额外的方言文本目录
    extra_texts = load_additional_dialect_corpus(additional_corpus_dir)
    
    # 合并所有文本
    all_texts = asr_texts + extra_texts
    
    if len(all_texts) == 0:
        print("错误: 没有可用的训练文本！")
        print("请确保已运行 1_prepare_dataset.py 或提供额外的方言语料。")
        return
    
    print(f"\n总共 {len(all_texts)} 条训练文本")
    
    # 3. 加载tokenizer和模型
    print(f"\n步骤 2: 加载 {lm_model_name} 模型...")
    tokenizer = BertTokenizer.from_pretrained(lm_model_name)
    model = BertForMaskedLM.from_pretrained(lm_model_name)
    
    print(f"模型参数量: {model.num_parameters():,}")
    
    # 4. 准备数据集
    print("\n步骤 3: 准备MLM数据集...")
    tokenized_dataset = prepare_mlm_dataset(
        all_texts,
        tokenizer,
        max_length=max_seq_length,
        test_size=0.1
    )
    
    print(f"训练集大小: {len(tokenized_dataset['train'])}")
    print(f"验证集大小: {len(tokenized_dataset['test'])}")
    
    # 5. 数据整理器（用于MLM）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15  # 15%的token会被mask
    )
    
    # 6. 训练参数
    print("\n步骤 4: 配置训练参数...")
    training_args = TrainingArguments(
        output_dir=lm_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        
        # 评估和保存
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # 日志
        logging_steps=100,
        logging_dir=f"{lm_output_dir}/logs",
        report_to=["tensorboard"],
        
        # 其他
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        
        # 推送到Hub（可选）
        push_to_hub=False,
    )
    
    print(f"训练配置:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  FP16: {training_args.fp16}")
    
    # 7. 初始化Trainer
    print("\n步骤 5: 初始化Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )
    
    # 8. 开始训练
    print("\n" + "=" * 80)
    print("开始训练...")
    print("=" * 80 + "\n")
    
    train_result = trainer.train()
    
    # 9. 保存模型
    print("\n" + "=" * 80)
    print("训练完成！保存模型...")
    print("=" * 80)
    
    trainer.save_model()
    tokenizer.save_pretrained(lm_output_dir)
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # 最终评估
    print("\n进行最终评估...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    print("\n" + "=" * 80)
    print("语言模型微调完成！")
    print("=" * 80)
    print(f"\n最终评估损失: {eval_metrics['eval_loss']:.4f}")
    print(f"\n模型保存位置: {lm_output_dir}")
    print("\n使用方法:")
    print("  from transformers import BertForMaskedLM, BertTokenizer")
    print(f"  model = BertForMaskedLM.from_pretrained('{lm_output_dir}')")
    print(f"  tokenizer = BertTokenizer.from_pretrained('{lm_output_dir}')")
    print("\n该模型可用于:")
    print("  1. ASR结果的语言模型评分")
    print("  2. 方言文本的生成和纠错")
    print("  3. N-best候选的重排序")


if __name__ == "__main__":
    main()

