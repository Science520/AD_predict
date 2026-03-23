"""
Whisper模型LoRA微调脚本 (官方标准实现)
功能：
1. 加载Whisper-large-v3模型
2. 添加自定义方言特殊token
3. 配置LoRA参数高效微调
4. 严格遵循Hugging Face官方最佳实践
5. 训练并保存LoRA适配器

参考: https://huggingface.co/blog/fine-tune-whisper
"""

import os
import sys
import yaml
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# 设置Hugging Face环境变量（使用镜像和离线模式）
os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'  # 离线模式

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk, Audio

# 确保导入installed evaluate package，而不是本地evaluate.py
import importlib
_sys_path_backup = sys.path.copy()
sys.path = [p for p in sys.path if not p.endswith('/scripts') and not p.endswith('/AD_predict')]
evaluate = importlib.import_module('evaluate')
sys.path = _sys_path_backup
del _sys_path_backup


def load_config(config_path="configs/training_args.yaml"):
    """加载配置文件"""
    # 获取项目根目录（脚本所在目录的父目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 如果config_path是相对路径，则相对于项目根目录
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    官方标准DataCollator - 完全遵循Hugging Face教程
    参考: https://huggingface.co/blog/fine-tune-whisper
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        # ⚠️ 关键修复：只保留Whisper需要的键，移除任何可能导致PEFT混淆的键
        # PEFT会自动查找input_ids，但Whisper用的是input_features
        return {
            "input_features": batch["input_features"],
            "labels": batch["labels"]
        }


def prepare_dataset(batch, feature_extractor, tokenizer):
    """
    官方标准数据准备函数（使用soundfile手动加载音频）
    参考: https://huggingface.co/blog/fine-tune-whisper
    """
    import soundfile as sf
    import librosa
    import numpy as np
    import os
    
    # 使用audio_path字段（已在前面的map步骤中提取）
    audio_path = batch["audio_path"]
    
    # 如果是相对路径，转为绝对路径
    if not os.path.isabs(audio_path) and not os.path.exists(audio_path):
        # 尝试在多个可能的目录中查找
        base_dirs = [
            '/data/AD_predict/data/raw/audio/elderly_audios',  # 原始音频
            '/data/AD_predict/data/raw/audio/elderly_audios_augmented',  # 增强音频
            os.path.expanduser('~/AD_predict/AD_predict/data/raw/audio/elderly_audios'),
            os.path.expanduser('~/AD_predict/AD_predict/data/raw/audio/elderly_audios_augmented'),
        ]
        for base_dir in base_dirs:
            full_path = os.path.join(base_dir, audio_path)
            if os.path.exists(full_path):
                audio_path = full_path
                break
    
    try:
        # 使用soundfile加载音频
        audio_array, sample_rate = sf.read(audio_path)
        
        # 如果是立体声，转换为单声道
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        # 重采样到16kHz（如果需要）
        if sample_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
            
    except Exception as e:
        # 静默处理缺失文件，使用空数据（避免大量警告信息）
        # 注意：这会影响训练质量，建议重新运行数据准备脚本
        audio_array = np.zeros(16000)
        sample_rate = 16000

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(
        audio_array, 
        sampling_rate=sample_rate
    ).input_features[0]

    # encode target text to label ids 
    # ⚠️ 关键修复：截断过长的标签序列（Whisper max_length=448）
    labels = tokenizer(
        batch["text"],
        max_length=448,
        truncation=True,
        return_tensors=None  # 返回Python list
    ).input_ids
    
    batch["labels"] = labels
    
    return batch


def compute_metrics(pred, tokenizer):
    """
    计算WER (Word Error Rate)
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    # 确保返回float而不是tensor（避免DataParallel gather警告）
    return {"wer": float(wer)}


def add_dialect_tokens(processor, dialect_labels):
    """
    添加方言特殊token到tokenizer
    """
    tokenizer = processor.tokenizer
    special_tokens = [f"<|dialect:{label}|>" for label in dialect_labels]
    
    print(f"\n添加 {len(special_tokens)} 个方言特殊token:")
    for token in special_tokens:
        print(f"  {token}")
    
    # 添加特殊token
    num_added = tokenizer.add_tokens(special_tokens)
    print(f"\n成功添加 {num_added} 个新token")
    print(f"词表大小: {len(tokenizer)} (原始: {len(tokenizer) - num_added})")
    
    return processor


def main(resume_from_checkpoint=None, config_path="configs/training_args.yaml"):
    """主函数
    
    Args:
        resume_from_checkpoint: 从指定checkpoint恢复训练，可以是：
            - None: 从头开始训练
            - "auto": 自动查找最新checkpoint
            - "path/to/checkpoint": 指定checkpoint路径
        config_path: 配置文件路径（默认: configs/training_args.yaml）
    """
    print("=" * 80)
    print("Whisper LoRA 微调 (官方标准实现)")
    print("=" * 80)
    print(f"配置文件: {config_path}")
    
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 1. 加载配置
    config = load_config(config_path)
    
    model_name = config['model']['name']
    output_dir = config['training']['output_dir']
    processed_data_dir = config['data']['processed_data_dir']
    dialect_labels = config['dialects']['labels']
    
    # 将相对路径转换为绝对路径（相对于项目根目录）
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    if not os.path.isabs(processed_data_dir):
        processed_data_dir = os.path.join(project_root, processed_data_dir)
    
    print(f"\n模型: {model_name}")
    print(f"数据目录: {processed_data_dir}")
    print(f"输出目录: {output_dir}")
    
    # 2. 加载预处理数据
    print("\n步骤 1: 加载预处理数据集...")
    dataset = load_from_disk(processed_data_dir)
    print(f"训练集大小: {len(dataset['train'])}")
    print(f"验证集大小: {len(dataset['validation'])}")
    
    # 3. 加载Whisper模型和处理器
    print("\n步骤 2: 加载Whisper模型和处理器...")
    processor = WhisperProcessor.from_pretrained(
        model_name, 
        language="zh", 
        task="transcribe"
    )
    
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        use_cache=False  # 训练时禁用缓存
    )
    
    # 启用gradient checkpointing以节省内存
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    print("✓ Gradient checkpointing已启用")
    
    # 额外的内存优化
    # 🔥 强制禁用多GPU和DataParallel
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # CUDA_VISIBLE_DEVICES应该在启动脚本中设置
    
    torch.cuda.empty_cache()  # 清理GPU缓存
    
    # GPU设备选择逻辑修复
    # 如果设置了CUDA_VISIBLE_DEVICES=1，则cuda:0就是物理GPU 1
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"✓ 可用GPU数量: {num_gpus}")
        
        if num_gpus > 1:
            print("⚠️ 检测到多个GPU，但将只使用第一个GPU以节省内存")
        
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    总内存: {props.total_memory / 1024**3:.2f} GB")
        
        # 使用第一个可见的GPU（如果CUDA_VISIBLE_DEVICES=1，这就是物理GPU 1）
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        
        # 更激进的内存限制
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.95, device=0)
        
        print(f"✓ 使用 cuda:0 (对应CUDA_VISIBLE_DEVICES指定的GPU)")
        print(f"✓ 内存优化设置已应用")
    else:
        device = None
        print("⚠️ CUDA不可用，使用CPU训练")
    
    # 设置语言和任务
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    # 4. 添加方言特殊token
    print("\n步骤 3: 添加方言特殊token...")
    processor = add_dialect_tokens(processor, dialect_labels)
    
    # 调整模型词嵌入大小以适应新token
    model.resize_token_embeddings(len(processor.tokenizer))
    print(f"模型词嵌入已调整为: {len(processor.tokenizer)}")
    
    # 5. 配置LoRA
    print("\n步骤 4: 配置LoRA...")
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias="none",
        # ⚠️ 关键修复：不指定task_type，让PEFT使用通用包装
        # task_type="SEQ_2_SEQ_LM" 会导致PEFT假设输入是input_ids
        # 但Whisper的编码器输入是input_features（音频特征）
    )
    
    print(f"LoRA配置:")
    print(f"  r (秩): {lora_config.r}")
    print(f"  lora_alpha: {lora_config.lora_alpha}")
    print(f"  target_modules: {lora_config.target_modules}")
    print(f"  lora_dropout: {lora_config.lora_dropout}")
    
    # 应用LoRA到模型
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 6. 准备数据 - 使用.map进行预处理（手动加载音频，避开torchcodec）
    print("\n步骤 5: 准备训练数据...")
    print("使用soundfile/librosa手动加载音频（避开torchcodec依赖）")
    
    # 直接从Arrow表中提取audio路径，避免Feature系统的自动解码
    print("提取音频路径...")
    def add_audio_path_column(split_dataset):
        # 直接访问底层Arrow表获取path
        audio_column = split_dataset.data.column('audio')
        paths = []
        for i in range(len(audio_column)):
            audio_dict = audio_column[i].as_py()
            paths.append(audio_dict['path'] if isinstance(audio_dict, dict) else audio_dict)
        
        # 使用add_column方法
        return split_dataset.add_column('audio_path', paths)
    
    dataset['train'] = add_audio_path_column(dataset['train'])
    dataset['validation'] = add_audio_path_column(dataset['validation'])
    
    def prepare_dataset_wrapper(batch):
        return prepare_dataset(
            batch, 
            processor.feature_extractor, 
            processor.tokenizer
        )
    
    # 使用map处理数据集 - 禁用multiprocessing，手动加载音频
    print("处理训练集和验证集...")
    dataset = dataset.map(
        prepare_dataset_wrapper,
        remove_columns=dataset.column_names["train"],
        num_proc=None,  # 禁用multiprocessing（避免序列化问题）
        desc="预处理音频"
    )
    
    print(f"训练集列名: {dataset['train'].column_names}")
    print(f"验证集列名: {dataset['validation'].column_names}")
    
    # 7. 初始化DataCollator (官方标准实现)
    print("\n步骤 6: 初始化DataCollator...")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id
    )
    
    # 8. 配置训练参数
    print("\n步骤 7: 配置训练参数...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        optim=config['training']['optim'],
        weight_decay=config['training']['weight_decay'],
        eval_strategy=config['training']['evaluation_strategy'],
        eval_steps=config['training']['eval_steps'],
        save_strategy=config['training']['save_strategy'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        greater_is_better=config['training']['greater_is_better'],
        logging_steps=config['training']['logging_steps'],
        report_to=config['training']['report_to'],
        fp16=config['training']['fp16'],
        dataloader_num_workers=config['training'].get('dataloader_num_workers', 0),
        remove_unused_columns=config['training']['remove_unused_columns'],
        predict_with_generate=config['training']['predict_with_generate'],
        generation_max_length=config['training']['generation_max_length'],
        push_to_hub=False,
        # 🔥 强制单GPU训练，禁用DataParallel
        dataloader_pin_memory=config['training'].get('dataloader_pin_memory', False),
        ddp_find_unused_parameters=False,
        local_rank=-1,  # 禁用分布式训练
        # 🔥 梯度检查点（如果配置文件中指定）
        gradient_checkpointing=config['training'].get('gradient_checkpointing', False),
    )
    
    print(f"训练配置:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  FP16: {training_args.fp16}")
    
    # 9. 定义评估函数
    def compute_metrics_wrapper(pred):
        return compute_metrics(pred, processor.tokenizer)
    
    # 10. 初始化官方Seq2SeqTrainer
    print("\n步骤 8: 初始化Trainer...")
    
    # ⚠️ 关键修复：明确告诉Trainer label列的名称
    # 这样Trainer就不会自动推断input_ids等错误的列名
    training_args.label_names = ["labels"]
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper,
        tokenizer=processor.feature_extractor,
    )
    
    print("\n注意: 数据不平衡问题建议通过数据增强在 '1_prepare_dataset.py' 中解决")
    
    # 11. 检查是否需要从checkpoint恢复
    import glob
    
    if resume_from_checkpoint == "auto":
        # 自动查找最新的checkpoint
        checkpoints = sorted(
            glob.glob(os.path.join(output_dir, "checkpoint-*")),
            key=lambda x: int(x.split("-")[-1])
        )
        if checkpoints:
            resume_from_checkpoint = checkpoints[-1]
            print(f"\n🔄 自动检测到checkpoint: {os.path.basename(resume_from_checkpoint)}")
        else:
            resume_from_checkpoint = None
            print("\n✓ 未找到checkpoint，将从头开始训练")
    elif resume_from_checkpoint:
        print(f"\n🔄 将从checkpoint恢复训练: {resume_from_checkpoint}")
    
    # 12. 开始训练
    print("\n" + "=" * 80)
    if resume_from_checkpoint:
        print("继续训练...")
    else:
        print("开始训练...")
    print("=" * 80 + "\n")
    
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # 12. 保存LoRA适配器
    print("\n" + "=" * 80)
    print("训练完成！保存模型...")
    print("=" * 80)
    
    adapter_output_dir = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(adapter_output_dir)
    processor.save_pretrained(adapter_output_dir)
    
    print(f"\n✅ LoRA适配器已保存到: {adapter_output_dir}")
    
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
    print("微调完成！")
    print("=" * 80)
    print(f"\n最终WER: {eval_metrics['eval_wer']:.4f}")
    print(f"\nLoRA适配器保存位置: {adapter_output_dir}")
    print("\n使用方法:")
    print("  from transformers import WhisperForConditionalGeneration, WhisperProcessor")
    print("  from peft import PeftModel")
    print(f"  base_model = WhisperForConditionalGeneration.from_pretrained('{model_name}')")
    print(f"  model = PeftModel.from_pretrained(base_model, '{adapter_output_dir}')")
    print(f"  processor = WhisperProcessor.from_pretrained('{adapter_output_dir}')")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Whisper LoRA微调脚本")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从checkpoint恢复训练。可选值: 'auto'(自动查找最新), checkpoint路径, 或None(从头开始)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_args.yaml",
        help="配置文件路径（默认: configs/training_args.yaml）"
    )
    
    args = parser.parse_args()
    main(resume_from_checkpoint=args.resume_from_checkpoint, config_path=args.config)
