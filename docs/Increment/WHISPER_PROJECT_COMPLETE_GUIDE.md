# Whisper多方言老年人语音识别完整项目指南

## 📋 项目概述

本项目提供了一个完整的Whisper模型微调解决方案，专门针对中国多地区老年人方言语音数据集。项目包含两个主要部分：

1. **智能数据收集系统** - 自动采样、下载、标注方言数据
2. **LoRA微调系统** - 参数高效微调Whisper模型

---

## 🗂️ 完整文件结构

```
AD_predict/
├── scripts/
│   ├── whisper_data_collection/          # 📦 数据收集模块（新增）
│   │   ├── 0_run_data_pipeline.sh       # 完整数据收集流程
│   │   ├── 1_analyze_and_sample.py      # 智能采样分析
│   │   ├── 2_selective_download.py      # 选择性下载
│   │   ├── 3_scrape_subtitles.py        # 字幕爬取和标注
│   │   └── README.md                    # 数据收集文档
│   │
│   ├── 0_validate_data.py               # 数据验证
│   ├── 1_prepare_dataset.py             # 数据预处理
│   ├── 2_finetune_whisper_lora.py       # Whisper LoRA微调
│   ├── 3_finetune_lm.py                 # 语言模型微调（可选）
│   ├── 4_inference_test.py              # 推理测试
│   └── run_pipeline.sh                  # 完整训练流程
│
├── data_utils/
│   ├── __init__.py
│   └── audio_augment.py                 # 音频增强工具
│
├── configs/
│   └── training_args.yaml               # 训练配置
│
├── requirements_whisper.txt             # 项目依赖
├── WHISPER_FINETUNING_README.md        # 微调详细文档
├── QUICKSTART.md                        # 快速开始指南
└── WHISPER_PROJECT_COMPLETE_GUIDE.md   # 本文件
```

---

## 🚀 完整工作流程

### 阶段1: 智能数据收集 🔄

**目标**：从Bilibili收集并标注方言语音数据

```bash
# 进入项目目录
cd /home/saisai/AD_predict/AD_predict

# 方式1: 一键运行完整数据收集流程（推荐）
bash scripts/whisper_data_collection/0_run_data_pipeline.sh

# 方式2: 分步执行（更可控）
# 步骤1: 分析并生成采样计划
python scripts/whisper_data_collection/1_analyze_and_sample.py

# 步骤2: 选择性下载视频和提取音频
python scripts/whisper_data_collection/2_selective_download.py

# 步骤3: 爬取字幕并生成转录文本
python scripts/whisper_data_collection/3_scrape_subtitles.py
```

**输出文件**：
- `data/sampling_plan.json` - 采样计划
- `data/raw/audio/elderly_audios/*.wav` - 音频文件（16kHz）
- `data/raw/audio/result/test*.txt` - 转录文本
- `data/download_results.json` - 下载统计
- `data/transcript_results.json` - 转录统计

**关键特性**：
- ✅ 智能分析方言分布，自动计算每个方言需要的样本数
- ✅ 只下载需要的视频，节省带宽和存储
- ✅ 优先使用Bilibili字幕，支持Whisper备用转录
- ✅ 自动关联方言标签

### 阶段2: 数据验证和预处理 📊

**目标**：验证数据完整性并准备训练数据

```bash
# 步骤1: 验证数据
python scripts/0_validate_data.py

# 步骤2: 数据预处理
python scripts/1_prepare_dataset.py
```

**输出文件**：
- `data_validation_report.txt` - 数据验证报告
- `processed_data/train/` - 训练集
- `processed_data/validation/` - 验证集
- `processed_data/dataset_stats.json` - 数据统计
- `temp_augmented_audios/` - 数据增强音频（如果启用）

**关键特性**：
- ✅ 自动检查音频和文本文件完整性
- ✅ 分析方言分布和数据平衡性
- ✅ 对少数类别自动进行数据增强（4倍）
- ✅ 添加方言提示token到文本

### 阶段3: Whisper模型微调 🎯

**目标**：使用LoRA高效微调Whisper模型

```bash
# 微调Whisper模型
python scripts/2_finetune_whisper_lora.py

# 监控训练（另开终端）
tensorboard --logdir=./whisper_lora_dialect/logs
```

**输出文件**：
- `whisper_lora_dialect/final_adapter/` - LoRA适配器（核心产出）
- `whisper_lora_dialect/checkpoint-*/` - 训练检查点
- `whisper_lora_dialect/logs/` - TensorBoard日志

**关键特性**：
- ✅ LoRA参数高效微调（只训练0.3%的参数）
- ✅ 自动添加12个方言特殊token
- ✅ 支持FP16混合精度训练
- ✅ 自动评估WER（词错误率）

### 阶段4: 模型测试和部署 🧪

**目标**：测试微调后的模型性能

```bash
# 单文件测试
python scripts/4_inference_test.py --audio test.wav

# 批量测试
python scripts/4_inference_test.py --audio ./test_audios/ --output results.json

# 指定方言
python scripts/4_inference_test.py --audio test.wav --dialect beijing_mandarin
```

**可选：微调语言模型**

```bash
# 微调中文BERT用于后处理
python scripts/3_finetune_lm.py
```

---

## 🎯 快速开始（完整流程）

### 环境准备（约5分钟）

```bash
# 1. 创建虚拟环境
conda create -n whisper_ft python=3.10
conda activate whisper_ft

# 2. 安装Python依赖
pip install -r requirements_whisper.txt

# 3. 安装系统依赖
# Ubuntu/Debian:
apt-get install ffmpeg

# macOS:
brew install ffmpeg

# 4. 安装视频下载工具
pip install you-get
```

### 数据收集（时间取决于网速和视频数量）

```bash
# 一键运行数据收集流程
bash scripts/whisper_data_collection/0_run_data_pipeline.sh

# 流程会提示：
# 1. 是否继续下载？
# 2. 下载全部还是测试模式？
# 3. 是否使用Whisper备选转录？
# 4. 选择Whisper模型大小
```

### 模型训练（2-6小时，取决于GPU）

```bash
# 一键运行训练流程
bash scripts/run_pipeline.sh

# 或分步执行：
python scripts/0_validate_data.py
python scripts/1_prepare_dataset.py
python scripts/2_finetune_whisper_lora.py
```

### 测试模型（约10秒/文件）

```bash
python scripts/4_inference_test.py --audio your_audio.wav
```

---

## 📊 数据收集系统详解

### 智能采样策略

系统支持3种平衡策略：

1. **加权策略（weighted）** - 推荐 ⭐
   - 保留原始分布特点
   - 确保每个方言至少有最小样本数
   - 适合真实场景

2. **均匀策略（uniform）**
   - 每个方言样本数相同
   - 适合学术研究

3. **按比例策略（proportional）**
   - 严格按原始比例扩展
   - 保持分布不变

### 字幕获取策略

```
┌─────────────────┐
│  Bilibili字幕    │ ← 优先（速度快，质量好）
└────────┬────────┘
         │ 如果没有
         ▼
┌─────────────────┐
│  Whisper转录     │ ← 备选（需要GPU，较慢）
└─────────────────┘
```

### 数据增强技术

对样本少的方言类别，自动应用：

1. **时间拉伸**：速度变化 0.95-1.05x
2. **音调变化**：±1个半音
3. **高斯噪声**：轻微背景噪声
4. **时间偏移**：音频时间偏移

---

## ⚙️ 配置调优

### 数据收集配置

编辑 `scripts/whisper_data_collection/1_analyze_and_sample.py`:

```python
# 每个方言最少样本数
min_samples_per_dialect = 30  # 建议：20-50

# 平衡策略
balance_strategy = 'weighted'  # 推荐：weighted

# 目标总样本数
target_total_samples = None  # None表示自动计算
```

### 训练配置

编辑 `configs/training_args.yaml`:

```yaml
# 数据增强
data:
  augmentation:
    enable: true
    min_samples_threshold: 10
    augmentation_factor: 4

# LoRA参数
lora:
  r: 16              # 建议：8, 16, 32
  lora_alpha: 32     # 建议：r的2倍

# 训练参数
training:
  num_train_epochs: 10
  learning_rate: 1.0e-4
  per_device_train_batch_size: 8  # GPU内存不足可减小
```

---

## 📈 预期结果

### 数据收集

- **采样效率**：只下载需要的视频，节省 50-70% 带宽
- **标注准确率**：方言标签准确率 100%（来自Excel）
- **字幕覆盖率**：Bilibili字幕 40-60%，Whisper补充剩余

### 模型性能

- **WER（词错误率）**：
  - Baseline（未微调）：~0.30
  - 微调后：~0.15-0.25（取决于数据质量）
  
- **方言识别准确率**：
  - 常见方言（>100样本）：85-95%
  - 少数方言（<20样本）：60-75%

### 训练效率

- **模型大小**：
  - 基础模型：~3GB
  - LoRA适配器：~50MB（只需保存这个）
  
- **训练时间**（NVIDIA A100）：
  - 100样本：~30分钟
  - 500样本：~2小时
  - 1000样本：~4小时

---

## 🔧 故障排除

### 数据收集问题

**Q: you-get下载失败**
```bash
# 解决：更新you-get并使用cookies
pip install --upgrade you-get
python scripts/whisper_data_collection/2_selective_download.py --cookies cookies.txt
```

**Q: 某方言可用视频不足**
```python
# 解决：调整最小样本数
min_samples_per_dialect = 15  # 降低要求
```

**Q: Whisper转录太慢**
```bash
# 解决：使用更小的模型
python scripts/whisper_data_collection/3_scrape_subtitles.py --whisper_model base
```

### 训练问题

**Q: CUDA内存不足**
```yaml
# 解决：减小批次大小
training:
  per_device_train_batch_size: 4  # 从8改为4
  gradient_accumulation_steps: 4  # 从2改为4
```

**Q: WER过高**
```yaml
# 解决：调整超参数
training:
  num_train_epochs: 15  # 增加训练轮数
  learning_rate: 5.0e-5  # 降低学习率
lora:
  r: 32  # 增大LoRA秩
```

---

## 📚 文档索引

- **快速开始**: `QUICKSTART.md`
- **详细微调指南**: `WHISPER_FINETUNING_README.md`
- **数据收集指南**: `scripts/whisper_data_collection/README.md`
- **本完整指南**: `WHISPER_PROJECT_COMPLETE_GUIDE.md`

---

## 🎓 最佳实践

### 1. 数据收集阶段

✅ **建议做的**：
- 先运行小规模测试（5个视频）
- 定期检查下载结果和字幕质量
- 备份采样计划和下载结果JSON

❌ **避免做的**：
- 一次性下载所有视频（浪费资源）
- 跳过数据验证步骤
- 忽略方言分布不均衡问题

### 2. 训练阶段

✅ **建议做的**：
- 使用TensorBoard监控训练
- 保存多个检查点
- 在验证集上评估WER

❌ **避免做的**：
- 过拟合（训练轮数过多）
- 忽略GPU内存使用
- 不检查linter错误

### 3. 部署阶段

✅ **建议做的**：
- 在真实数据上测试
- 记录每个方言的性能
- 合并LoRA权重（如果需要加速推理）

❌ **避免做的**：
- 直接部署到生产环境（需先充分测试）
- 忽略模型的局限性
- 不监控推理性能

---

## 📞 获取帮助

1. **查看文档**：
   - 完整文档：`WHISPER_FINETUNING_README.md`
   - 快速指南：`QUICKSTART.md`
   - 数据收集：`scripts/whisper_data_collection/README.md`

2. **检查日志**：
   - 训练日志：`whisper_lora_dialect/logs/`
   - 数据验证报告：`data_validation_report.txt`

3. **查看结果**：
   - 采样计划：`data/sampling_plan.json`
   - 下载结果：`data/download_results.json`
   - 转录结果：`data/transcript_results.json`

---

## 🎉 项目亮点

### 创新点

1. **智能数据采集**：
   - 自动分析和优化采样策略
   - 避免盲目下载所有数据

2. **方言提示机制**：
   - 通过特殊token告知模型方言类型
   - 提升方言识别准确率

3. **数据不均衡处理**：
   - 自动数据增强
   - 支持加权采样

4. **参数高效微调**：
   - LoRA只训练0.3%参数
   - 适配器仅50MB

### 技术栈

- **深度学习**: PyTorch, Transformers, PEFT
- **语音处理**: Whisper, Librosa, FFmpeg
- **数据处理**: Pandas, Datasets, Audiomentations
- **网络爬虫**: Requests, BeautifulSoup4, you-get
- **工具**: TensorBoard, YAML

---

## 📅 开发路线图

- [x] 智能数据收集系统
- [x] LoRA微调框架
- [x] 数据增强
- [x] 方言提示机制
- [x] 完整文档
- [ ] 多GPU训练支持
- [ ] 模型蒸馏
- [ ] Web演示界面
- [ ] Docker部署

---

## 📄 许可证

MIT License

---

**🚀 开始您的Whisper方言微调之旅！**

有任何问题欢迎查阅文档或提issue。祝训练顺利！🎊

