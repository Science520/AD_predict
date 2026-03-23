# Whisper LoRA微调 - 最终实现说明

## 📋 概述

本文档记录了Whisper-large-v3模型LoRA微调的最终实现方案，严格遵循**Hugging Face官方最佳实践**。

## ✅ 核心实现（符合官方标准）

### 1. 数据预处理函数
```python
def prepare_dataset(batch, feature_extractor, tokenizer):
    # 使用soundfile手动加载音频
    audio_array, sample_rate = sf.read(audio_path)
    
    # 提取log-Mel特征
    batch["input_features"] = feature_extractor(audio_array, sampling_rate=16000).input_features[0]
    
    # 编码文本标签
    batch["labels"] = tokenizer(batch["text"]).input_ids
    
    return batch
```

### 2. 官方DataCollator
```python
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        # 分别处理input_features和labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 将padding替换为-100
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 删除bos token
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
```

### 3. 标准Seq2SeqTrainer
```python
trainer = Seq2SeqTrainer(
    model=model,  # PEFT包装的模型
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    data_collator=data_collator,  # 官方DataCollator
    compute_metrics=compute_metrics_wrapper,
    tokenizer=processor.feature_extractor,
)
```

## 🔧 解决的关键技术问题

### 问题1: Torchcodec依赖冲突 ⚠️

**现象**: 最新的datasets库要求torchcodec解码音频，但torchcodec与PyTorch 2.7.1不兼容。

**解决方案**:
1. **绕过Feature系统自动解码**
   ```python
   # 直接从Arrow表提取路径，不触发自动解码
   audio_column = split_dataset.data.column('audio')
   paths = [audio_column[i].as_py()['path'] for i in range(len(audio_column))]
   dataset = dataset.add_column('audio_path', paths)
   ```

2. **手动加载音频**
   ```python
   import soundfile as sf
   import librosa
   
   audio_array, sample_rate = sf.read(audio_path)
   if sample_rate != 16000:
       audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
   ```

### 问题2: 相对路径处理

**解决方案**: 动态获取项目根目录，所有相对路径相对于项目根目录解析。

```python
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
```

### 问题3: 参数兼容性

**解决方案**: 使用新版API参数名
- `evaluation_strategy` → `eval_strategy`
- 确保DataCollator返回的batch只包含`input_features`和`labels`

## 📊 训练配置

### LoRA参数
- **r**: 16 (秩)
- **lora_alpha**: 32
- **target_modules**: ["q_proj", "v_proj"]
- **lora_dropout**: 0.05
- **task_type**: SEQ_2_SEQ_LM

### 训练参数
- **Epochs**: 10
- **Batch size**: 8 (per device)
- **Gradient accumulation**: 2
- **Learning rate**: 1e-4
- **FP16**: True
- **优化器**: AdamW

## 🎯 方言支持

支持12个中国方言：
- 北京官话 (beijing_mandarin)
- 江淮官话 (jianghuai_mandarin)
- 中原官话 (zhongyuan_mandarin)
- 东北官话 (dongbei_mandarin)
- 晋语 (jin_dialect)
- 兰银官话 (lanyin_mandarin)
- 吴语 (wu_dialect)
- 西南官话 (xinan_mandarin)
- 闽语 (min_dialect)
- 粤语 (yue_dialect)
- 藏语 (tibetan_dialect)
- 赣语 (gan_dialect)

## 📁 项目结构

```
AD_predict/
├── scripts/
│   ├── 1_prepare_dataset.py       # 数据准备
│   └── 2_finetune_whisper_lora.py # 训练脚本（最终版）
├── configs/
│   └── training_args.yaml          # 训练配置
├── processed_data/                 # 预处理数据
└── whisper_lora_dialect/           # 输出目录
    └── final_adapter/              # LoRA适配器权重
```

## 🚀 运行方式

```bash
# 从项目根目录运行
cd /home/saisai/AD_predict/AD_predict
conda activate graph
export HF_HOME=~/.cache/huggingface
export HF_ENDPOINT='https://hf-mirror.com'
export HF_HUB_OFFLINE=1
python scripts/2_finetune_whisper_lora.py
```

## 📝 重要提示

### ✅ 做到的
1. **100%遵循Hugging Face官方标准** - 使用官方DataCollator和Seq2SeqTrainer
2. **无自定义Trainer子类** - 完全依赖标准库
3. **解决环境兼容性** - 绕过torchcodec依赖
4. **支持从任何目录运行** - 动态路径解析

### ⚠️ 数据不平衡处理
数据不平衡问题应在`1_prepare_dataset.py`中通过**数据增强**解决，而不是在训练脚本中使用加权采样。

## 📚 参考资料

- [Hugging Face Whisper Fine-tuning Guide](https://huggingface.co/blog/fine-tune-whisper)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Seq2SeqTrainer](https://huggingface.co/docs/transformers/main_classes/trainer)

## 🏆 可训练参数

```
trainable params: 7,864,320 || all params: 1,551,370,240 || trainable%: 0.51%
```

仅训练模型参数的0.51%，大幅降低训练成本！

---

**最后更新**: 2025-10-18
**状态**: ✅ 可正常运行

