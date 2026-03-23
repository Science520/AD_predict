# Whisper多方言老年人语音识别微调项目

这是一个完整的Whisper模型LoRA微调项目，专门针对中国多地区老年人方言语音数据集进行优化。

## 📁 项目结构

```
AD_predict/
├── scripts/
│   ├── 1_prepare_dataset.py       # 数据预处理脚本
│   ├── 2_finetune_whisper_lora.py # Whisper LoRA微调脚本
│   └── 3_finetune_lm.py           # 语言模型微调脚本（可选）
├── data_utils/
│   └── audio_augment.py           # 音频增强工具
├── configs/
│   └── training_args.yaml         # 训练配置文件
├── requirements_whisper.txt       # 项目依赖
└── WHISPER_FINETUNING_README.md   # 本文件
```

## 🎯 核心特性

1. **方言标签注入**: 通过特殊token（如`<|dialect:beijing_mandarin|>`）告知模型音频的方言类型
2. **数据不均衡处理**: 
   - 对样本少的方言类别进行数据增强（变速、变调、加噪声等）
   - 支持加权采样（可选）
3. **LoRA参数高效微调**: 只训练少量参数，大幅降低计算和存储成本
4. **12个方言类别**: 
   - beijing_mandarin (北京官话)
   - wu_dialect (吴语)
   - dongbei_mandarin (东北官话)
   - zhongyuan_mandarin (中原官话)
   - lanyin_mandarin (兰银官话)
   - jianghuai_mandarin (江淮官话)
   - xinan_mandarin (西南官话)
   - jin_dialect (晋语)
   - yue_dialect (粤语)
   - gan_dialect (赣语)
   - min_dialect (闽语)
   - tibetan_dialect (藏语)

## 🚀 快速开始

### 步骤 1: 环境准备

```bash
# 创建虚拟环境（推荐）
conda create -n whisper_ft python=3.10
conda activate whisper_ft

# 安装依赖
pip install -r requirements_whisper.txt

# 如果遇到torch相关问题，请根据CUDA版本安装：
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 步骤 2: 数据预处理

```bash
python scripts/1_prepare_dataset.py
```

**功能说明**:
- 读取Excel文件 (`老人视频信息_final_complete_20251016_214400.xlsx`)
- 匹配音频文件和转录文本
- 为每条数据添加方言提示token
- 对样本少的方言类别进行4倍数据增强
- 按9:1比例分割训练集和验证集
- 保存到 `./processed_data` 目录

**输出**:
```
processed_data/
├── train/
├── validation/
└── dataset_stats.json
```

### 步骤 3: 微调Whisper模型

```bash
python scripts/2_finetune_whisper_lora.py
```

**功能说明**:
- 加载 `openai/whisper-large-v3` 模型
- 添加12个方言特殊token并扩展词表
- 配置LoRA (r=16, alpha=32, target=["q_proj", "v_proj"])
- 使用混合精度训练（FP16）
- 自动评估WER（Word Error Rate）
- 保存轻量级LoRA适配器（约几十MB）

**预期训练时间**:
- GPU: NVIDIA A100 (40GB) - 约2-4小时（取决于数据量）
- GPU: NVIDIA RTX 4090 - 约3-6小时
- GPU: NVIDIA RTX 3090 - 约4-8小时

**输出**:
```
whisper_lora_dialect/
├── final_adapter/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
├── checkpoint-500/
├── checkpoint-1000/
└── logs/
```

### 步骤 4: 微调语言模型（可选）

```bash
python scripts/3_finetune_lm.py
```

**功能说明**:
- 微调 `bert-base-chinese` 模型
- 使用ASR转录文本进行Masked Language Modeling
- 可用于后续的ASR结果重排序和纠错

## ⚙️ 配置文件说明

编辑 `configs/training_args.yaml` 来自定义训练参数：

```yaml
# 关键参数
data:
  augmentation:
    enable: true                    # 是否启用数据增强
    min_samples_threshold: 10       # 少于此值的类别将被增强
    augmentation_factor: 4          # 增强倍数

lora:
  r: 16                             # LoRA秩（越大参数越多）
  lora_alpha: 32                    # LoRA缩放因子
  target_modules: ["q_proj", "v_proj"]  # 目标模块

training:
  num_train_epochs: 10              # 训练轮数
  per_device_train_batch_size: 8   # 批次大小
  learning_rate: 1.0e-4             # 学习率
  fp16: true                        # 混合精度训练
```

## 📊 数据增强策略

针对样本数量少于10个的方言类别，系统会自动应用以下增强技术：

1. **时间拉伸** (Time Stretch): 速度变化 0.95-1.05倍
2. **音调变化** (Pitch Shift): 音高变化 ±1个半音
3. **高斯噪声** (Gaussian Noise): 轻微背景噪声
4. **时间偏移** (Time Shift): 音频时间偏移

## 🔧 使用训练好的模型

### 方法1: 使用脚本推理

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import torch

# 加载基础模型
base_model_name = "openai/whisper-large-v3"
processor = WhisperProcessor.from_pretrained(base_model_name, language="zh", task="transcribe")
base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name, torch_dtype=torch.float16)

# 加载LoRA适配器
model = PeftModel.from_pretrained(base_model, "./whisper_lora_dialect/final_adapter")
model.to("cuda")

# 推理
import librosa
audio, sr = librosa.load("test_audio.wav", sr=16000)
input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
input_features = input_features.to("cuda", torch.float16)

# 添加方言提示
dialect_token = "<|dialect:beijing_mandarin|>"
forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")

# 生成
predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print(f"转录结果: {transcription}")
```

### 方法2: 合并LoRA权重到基础模型

```python
from transformers import WhisperForConditionalGeneration
from peft import PeftModel

# 加载基础模型和适配器
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
model = PeftModel.from_pretrained(base_model, "./whisper_lora_dialect/final_adapter")

# 合并权重
merged_model = model.merge_and_unload()

# 保存完整模型
merged_model.save_pretrained("./whisper_dialect_merged")
```

## 📈 监控训练

### TensorBoard

```bash
tensorboard --logdir=./whisper_lora_dialect/logs
```

访问 `http://localhost:6006` 查看训练曲线。

### Weights & Biases（可选）

1. 在 `configs/training_args.yaml` 中设置 `report_to: ["wandb"]`
2. 运行前登录：`wandb login`

## 🐛 常见问题

### 1. CUDA内存不足

**解决方案**:
- 减小 `per_device_train_batch_size`（如从8改为4）
- 增大 `gradient_accumulation_steps`（如从2改为4）
- 使用更小的模型：`openai/whisper-medium` 或 `openai/whisper-small`

### 2. 数据加载失败

**检查**:
- Excel文件路径是否正确
- 音频文件是否存在
- 转录文本文件是否存在

### 3. 特殊token未被识别

**确保**:
- 在训练脚本中正确添加了方言token
- processor.tokenizer已保存并重新加载
- 模型词嵌入已调整大小

### 4. WER过高

**优化建议**:
- 增加训练轮数
- 调整学习率（尝试5e-5或2e-4）
- 增大LoRA的秩（r）
- 添加更多训练数据
- 检查数据质量

## 📝 数据要求

### Excel文件格式

| 列名 | 说明 | 示例 |
|-----|------|------|
| up主 | 上传者 | 闲聊北京 |
| 视频名称 | 视频标题 | 75岁铁路大爷退休金多少 |
| url | 视频URL | https://... |
| dialect_label | 方言标签 | beijing_mandarin |

### 音频要求

- 格式: WAV (推荐) 或 MP3
- 采样率: 16000 Hz (会自动重采样)
- 时长: 建议5-30秒
- 质量: 清晰，噪声较少

## 🔬 实验建议

### Baseline实验

1. **不使用数据增强**: 设置 `augmentation.enable: false`
2. **不使用方言提示**: 在`1_prepare_dataset.py`中注释掉方言token
3. **全参数微调 vs LoRA**: 对比训练效率和效果

### 超参数调优

建议尝试的参数组合：

| 参数 | 选项1 | 选项2 | 选项3 |
|-----|-------|-------|-------|
| learning_rate | 5e-5 | 1e-4 | 2e-4 |
| lora_r | 8 | 16 | 32 |
| lora_alpha | 16 | 32 | 64 |
| num_epochs | 5 | 10 | 15 |

## 📚 参考资源

- [Whisper论文](https://arxiv.org/abs/2212.04356)
- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers文档](https://huggingface.co/docs/transformers)
- [PEFT文档](https://huggingface.co/docs/peft)

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目遵循MIT许可证。

## 联系方式

如有问题，请联系项目维护者。

---

**祝训练顺利！🎉**

