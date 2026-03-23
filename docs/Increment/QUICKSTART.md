# 快速开始指南

本指南帮助您快速开始Whisper多方言微调项目。

## 🚀 5分钟快速开始

### 1. 安装依赖（约2-3分钟）

```bash
# 创建虚拟环境（推荐）
conda create -n whisper_ft python=3.10
conda activate whisper_ft

# 安装依赖
pip install -r requirements_whisper.txt
```

### 2. 验证数据（约30秒）

```bash
python scripts/0_validate_data.py
```

这会检查：
- ✅ Excel文件是否存在
- ✅ 音频文件是否可读
- ✅ 转录文本是否完整
- ✅ 方言分布是否均衡

### 3. 运行完整流程（可选：自动化）

```bash
# 一键运行所有步骤
bash scripts/run_pipeline.sh
```

或者**手动逐步执行**（推荐，更可控）：

#### 步骤A: 数据预处理（约5-10分钟）

```bash
python scripts/1_prepare_dataset.py
```

完成后会生成：
- `./processed_data/train/` - 训练集
- `./processed_data/validation/` - 验证集
- `./processed_data/dataset_stats.json` - 数据统计

#### 步骤B: 微调模型（约2-6小时，取决于GPU）

```bash
python scripts/2_finetune_whisper_lora.py
```

完成后会生成：
- `./whisper_lora_dialect/final_adapter/` - LoRA适配器（**核心产出**）
- `./whisper_lora_dialect/logs/` - 训练日志

#### 步骤C: 测试模型（约10秒/文件）

```bash
# 单文件测试
python scripts/4_inference_test.py --audio test.wav

# 批量测试
python scripts/4_inference_test.py --audio ./test_audios/ --output results.json

# 指定方言
python scripts/4_inference_test.py --audio test.wav --dialect beijing_mandarin
```

## 📝 自定义配置

编辑 `configs/training_args.yaml` 来调整参数：

```yaml
# 关键参数说明
data:
  augmentation:
    enable: true              # 是否数据增强（推荐开启）
    min_samples_threshold: 10 # 少于10个样本的类别会被增强

lora:
  r: 16                      # LoRA秩（8/16/32，越大参数越多）
  lora_alpha: 32             # 缩放因子（通常是r的2倍）

training:
  num_train_epochs: 10       # 训练轮数
  learning_rate: 1.0e-4      # 学习率
  per_device_train_batch_size: 8  # 批次大小（GPU内存不足可减小）
```

## 🔧 常见问题速查

### Q1: CUDA内存不足 (Out of Memory)

**方案1**: 减小批次大小
```yaml
# configs/training_args.yaml
training:
  per_device_train_batch_size: 4  # 从8改为4
  gradient_accumulation_steps: 4  # 从2改为4
```

**方案2**: 使用更小的模型
```yaml
# configs/training_args.yaml
model:
  name: "openai/whisper-medium"  # 或 whisper-small
```

### Q2: 找不到音频文件

检查配置文件中的路径：
```yaml
# configs/training_args.yaml
data:
  excel_path: "/path/to/your/excel.xlsx"
  audio_base_dir: "/path/to/audio/directory"
```

### Q3: 训练很慢

确保使用GPU：
```bash
# 检查GPU
nvidia-smi

# 检查PyTorch是否识别GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Q4: WER（词错误率）过高

尝试：
1. 增加训练轮数（10 -> 15）
2. 调整学习率（试试5e-5或2e-4）
3. 增大LoRA的r（16 -> 32）
4. 检查数据质量

## 📊 监控训练

### TensorBoard（推荐）

```bash
# 启动TensorBoard
tensorboard --logdir=./whisper_lora_dialect/logs

# 在浏览器访问
# http://localhost:6006
```

### Weights & Biases（可选）

```bash
# 安装
pip install wandb

# 登录
wandb login

# 修改配置
# configs/training_args.yaml
training:
  report_to: ["wandb"]
```

## 🎯 预期结果

训练成功后，您应该看到：

```
✅ LoRA适配器已保存到: ./whisper_lora_dialect/final_adapter

最终WER: 0.15-0.25（取决于数据质量）

使用方法:
  from transformers import WhisperForConditionalGeneration
  from peft import PeftModel
  
  base_model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3')
  model = PeftModel.from_pretrained(base_model, './whisper_lora_dialect/final_adapter')
```

## 📚 进阶使用

### 方言感知语言模型微调（可选）

```bash
python scripts/3_finetune_lm.py
```

### 推理优化

```python
# 合并LoRA权重到基础模型（加速推理）
from peft import PeftModel

base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
model = PeftModel.from_pretrained(base_model, "./whisper_lora_dialect/final_adapter")

# 合并
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./whisper_dialect_merged")
```

## 🆘 获取帮助

1. 查看详细文档：`WHISPER_FINETUNING_README.md`
2. 检查数据验证报告：`data_validation_report.txt`
3. 查看训练日志：`./whisper_lora_dialect/logs/`

## ✅ 检查清单

训练前确认：
- [ ] 已安装所有依赖
- [ ] GPU可用（推荐）
- [ ] 数据验证通过
- [ ] 配置文件已检查
- [ ] 有足够的磁盘空间（>10GB）

开始训练！🚀

