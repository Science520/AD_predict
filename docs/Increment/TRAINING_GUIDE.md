# 🎯 Whisper LoRA 训练完全指南

## 📦 模型文件保存位置

### 1. 训练过程中的文件
```
./whisper_lora_dialect/
├── checkpoint-500/          # 每500步保存一个checkpoint
│   ├── adapter_config.json  # LoRA配置
│   ├── adapter_model.bin    # LoRA权重
│   ├── trainer_state.json   # 训练状态
│   ├── training_args.bin    # 训练参数
│   └── ...
├── checkpoint-1000/
├── checkpoint-1500/
├── logs/                    # TensorBoard日志
│   └── events.out.tfevents...
└── runs/                    # 训练运行记录
```

### 2. 最终模型文件
```
./whisper_lora_dialect/
└── final_adapter/           # ✅ 最终的LoRA适配器
    ├── adapter_config.json  # LoRA配置
    ├── adapter_model.bin    # LoRA权重（通常2-30MB）
    ├── preprocessor_config.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── ...
```

**重要**: 
- LoRA只保存**适配器权重**（很小，通常<50MB）
- **不保存完整的whisper-large-v3模型**（3GB+）
- 使用时需要加载基础模型+适配器

---

## 🔄 Checkpoint配置说明

当前配置（`configs/training_args.yaml`）：

```yaml
# 每500步评估一次
eval_steps: 500

# 每500步保存一个checkpoint
save_strategy: "steps"
save_steps: 500

# 最多保留3个checkpoint（自动删除旧的）
save_total_limit: 3

# 训练结束时加载最佳模型
load_best_model_at_end: true

# 以WER作为评估指标（越小越好）
metric_for_best_model: "wer"
greater_is_better: false
```

---

## 🔧 如何从Checkpoint恢复训练

### 方法1: 自动恢复（推荐）

在训练脚本中添加恢复逻辑：

```python
# 修改 scripts/2_finetune_whisper_lora.py 的 main() 函数

# 在 trainer.train() 前添加：
import glob

# 检查是否有checkpoint
checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
resume_from_checkpoint = None

if checkpoints:
    # 找到最新的checkpoint
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    print(f"⚠️ 发现checkpoint: {latest_checkpoint}")
    response = input("是否从此checkpoint继续训练? (y/n): ")
    if response.lower() == 'y':
        resume_from_checkpoint = latest_checkpoint
        print(f"✓ 将从 {resume_from_checkpoint} 继续训练")

# 修改训练调用
train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
```

### 方法2: 手动指定

直接修改训练调用：

```python
# 在 scripts/2_finetune_whisper_lora.py 中
train_result = trainer.train(
    resume_from_checkpoint="./whisper_lora_dialect/checkpoint-1500"
)
```

### 方法3: 命令行参数（已实现）✅

```bash
# 从头开始训练
python scripts/2_finetune_whisper_lora.py

# 自动查找并从最新checkpoint恢复
python scripts/2_finetune_whisper_lora.py --resume_from_checkpoint auto

# 从指定checkpoint恢复
python scripts/2_finetune_whisper_lora.py --resume_from_checkpoint ./whisper_lora_dialect/checkpoint-1500
```

---

## 📊 监控训练进度

### 使用进度查看脚本

```bash
# 一次性查看当前进度
python scripts/check_training_progress.py

# 持续监控（每30秒刷新）
python scripts/check_training_progress.py --watch

# 自定义刷新间隔（每10秒）
python scripts/check_training_progress.py --watch --interval 10
```

### 使用TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir ./whisper_lora_dialect/logs

# 在浏览器打开: http://localhost:6006
```

---

## 🚀 使用训练好的模型

### 方法1: 加载LoRA适配器

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import torch

# 1. 加载基础模型
base_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3"
)

# 2. 加载LoRA适配器
model = PeftModel.from_pretrained(
    base_model, 
    "./whisper_lora_dialect/final_adapter"
)

# 3. 加载processor
processor = WhisperProcessor.from_pretrained(
    "./whisper_lora_dialect/final_adapter",
    language="zh",
    task="transcribe"
)

# 4. 推理
audio_file = "path/to/audio.wav"
# ... (推理代码)
```

### 方法2: 合并适配器到基础模型（可选）

```python
from peft import PeftModel

# 加载模型
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
model = PeftModel.from_pretrained(base_model, "./whisper_lora_dialect/final_adapter")

# 合并LoRA权重到基础模型
model = model.merge_and_unload()

# 保存合并后的完整模型（约3GB）
model.save_pretrained("./whisper_dialect_merged")
processor.save_pretrained("./whisper_dialect_merged")
```

---

## 📁 完整的文件结构

```
AD_predict/
├── configs/
│   └── training_args.yaml          # 训练配置
├── scripts/
│   ├── 1_prepare_dataset.py        # 数据准备
│   ├── 2_finetune_whisper_lora.py  # 训练脚本 ✅ 支持恢复
│   └── check_training_progress.py  # 进度查看
├── processed_data/                 # 预处理数据
│   ├── train/
│   └── validation/
├── whisper_lora_dialect/           # 训练输出 ⭐
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   ├── checkpoint-1500/            # 中间checkpoints
│   ├── final_adapter/              # ✅ 最终模型
│   │   ├── adapter_config.json
│   │   ├── adapter_model.bin       # LoRA权重（小）
│   │   └── ...
│   ├── logs/                       # TensorBoard日志
│   ├── train_results.json          # 训练指标
│   └── all_results.json            # 所有结果
└── whisper_training.log            # 训练日志
```

---

## ⚙️ 训练配置说明

当前配置（已优化内存使用）：

```yaml
per_device_train_batch_size: 2      # 批次大小（已减小）
per_device_eval_batch_size: 2
gradient_accumulation_steps: 8      # 梯度累积（增加以保持有效batch=16）
learning_rate: 1.0e-4
num_train_epochs: 10
fp16: true                          # 混合精度训练
```

**内存优化措施**：
- ✅ Gradient checkpointing已启用
- ✅ 小batch size + 梯度累积
- ✅ FP16混合精度训练
- ✅ LoRA只训练0.5%参数

---

## 🔍 故障排查

### 1. GPU内存不足

**症状**: `torch.OutOfMemoryError: CUDA out of memory`

**解决方案**:
```yaml
# 在 configs/training_args.yaml 中进一步减小batch size
per_device_train_batch_size: 1      # 从2改为1
gradient_accumulation_steps: 16     # 从8改为16
```

### 2. 训练中断

**解决方案**:
```bash
# 自动从最新checkpoint恢复
python scripts/2_finetune_whisper_lora.py --resume_from_checkpoint auto
```

### 3. 找不到音频文件

**症状**: `警告：无法加载音频 aug_xxx.wav`

**解决方案**:
```bash
# 重新运行数据准备脚本（已修复增强音频保存）
python scripts/1_prepare_dataset.py
```

---

## 📌 重要提示

1. **Checkpoint自动管理**: 系统会自动保留最近3个checkpoint（`save_total_limit: 3`）
2. **最佳模型保存**: 训练结束时会自动加载WER最低的模型
3. **LoRA权重小**: 适配器通常只有20-50MB，便于分享和部署
4. **基础模型**: 使用时仍需要基础模型`whisper-large-v3`（3GB）

---

## 📞 下一步

1. **重新准备数据**（修复增强音频保存）:
   ```bash
   python scripts/1_prepare_dataset.py
   ```

2. **开始训练**:
   ```bash
   python scripts/2_finetune_whisper_lora.py
   ```

3. **监控进度**:
   ```bash
   python scripts/check_training_progress.py --watch
   ```

4. **训练完成后**，模型保存在:
   ```
   ./whisper_lora_dialect/final_adapter/
   ```

