# 阿尔茨海默症多模态检测系统

一个基于概念瓶颈模型的可解释性阿尔茨海默症检测系统，融合语音和EEG信号进行诊断。

## 🎯 系统特点

- **多模态融合**：结合语音信号和EEG信号
- **可解释性**：基于概念瓶颈层的透明诊断过程
- **医学概念映射**：将底层特征转换为临床可理解的概念
- **端到端训练**：支持分阶段训练和端到端优化

## 🏗️ 系统架构

```
语音输入 → ASR模型(Wav2Vec2/Whisper) → 语音+文本特征
                                           ↓
EEG信号 → EEG特征提取器 → EEG特征 → 概念瓶颈层 → CRF分类器 → 诊断结果
                                           ↑
                                    文本精调模型
```

### 核心概念

**语音概念**：
- 语速 (Speech Rate)=从文本的词数和音频总时长计算
- 停顿比例 (Pause Ratio)=从文本中<pause>标记的数量和总次数计算
- 词汇丰富度 (Lexical Richness)=从文 本计算（如Type-Token）
- 句法复杂度 (Syntactic Complexity)=从文本计算（如依存句法树的深度）

**EEG概念**：
- Alpha波功率 (Alpha Power)
- Theta/Beta比值 (Theta-Beta Ratio)

## 🚀 快速开始

### 环境配置

```bash
# 创建conda环境
conda create -n alzheimer python=3.8
conda activate alzheimer

# 安装依赖
pip install -r requirements.txt

# 设置PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 数据准备

```bash
# 数据预处理
python scripts/preprocess_data.py \
    --config config/data_config.yaml \
    --data_csv data/dataset_manifest.csv \
    --output_dir data/processed
```

### 训练模型

```bash
# 分阶段训练 (推荐)
python scripts/train_asr.py --config config/model_config.yaml --experiment_name "asr_v1"
python scripts/train_concepts.py --config config/model_config.yaml --experiment_name "concepts_v1"
python scripts/train_crf.py --config config/model_config.yaml --experiment_name "crf_v1"

# 或端到端训练
python scripts/train_end2end.py --config config/model_config.yaml --experiment_name "e2e_v1"
```

### 模型推理

```bash
python scripts/inference.py \
    --model_path experiments/checkpoints/best_model.pth \
    --audio_path data/test/sample.wav \
    --eeg_path data/test/sample_eeg.csv \
    --output_html results/explanation.html
```

## 📁 项目结构

```
alzheimer_detection/
├── config/                 # 配置文件
├── src/                   # 源代码
│   ├── data/             # 数据处理
│   ├── models/           # 模型定义
│   ├── training/         # 训练相关
│   ├── inference/        # 推理相关
│   └── utils/           # 工具函数
├── scripts/              # 执行脚本
├── experiments/          # 实验管理
├── notebooks/           # Jupyter notebooks
└── tests/              # 单元测试
```

## 🔬 实验管理

项目集成了Weights & Biases进行实验跟踪，所有训练过程都会自动记录：
- 损失曲线
- 评估指标
- 概念预测准确率
- 模型可解释性分析

## 📊 可解释性

系统提供多层次的可解释性：
1. **概念级解释**：每个医学概念的预测值和置信度
2. **特征重要性**：各模态特征对最终诊断的贡献
3. **可视化报告**：生成HTML格式的详细解释报告

## 🧪 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_models.py -v
```



🔧 New Training & Evaluation Infrastructure
Scripts Created:
✅ scripts/train_end2end.py - Complete end-to-end training with multi-loss optimization
✅ scripts/train_concepts.py - Staged concept bottleneck layer training
✅ scripts/preprocess_data.py - Multi-modal data preprocessing pipeline
✅ scripts/inference.py - Model inference with explainable HTML reports
✅ scripts/evaluate.py - Comprehensive model evaluation and reporting
Utility Modules:
✅ src/utils/metrics.py - Classification & concept evaluation metrics
✅ src/utils/visualization.py - Training curves, confusion matrices, concept plots

🚀 How to Use Your System
1. Data Preprocessing:
```bash
python scripts/preprocess_data.py \
    --config config/data_config.yaml \
    --data_csv data/dataset_manifest.csv \
    --output_dir data/processed
```

**2. End-to-End Training:**
```bash
python scripts/train_end2end.py \
    --config config/model_config.yaml \
    --experiment_name "alzheimer_e2e_v1"
```

**3. Staged Training (Alternative):**
```bash
# Train concept bottleneck layer first
python scripts/train_concepts.py \
    --config config/model_config.yaml \
    --experiment_name "concepts_v1"

# Then full end-to-end training
python scripts/train_end2end.py \
    --config config/model_config.yaml \
    --experiment_name "e2e_v1"
```

**4. Model Inference:**
```bash
python scripts/inference.py \
    --model_path experiments/checkpoints/best_model.pth \
    --audio_path data/test/sample.wav \
    --eeg_path data/test/sample_eeg.csv \
    --output_html results/explanation.html
```

**5. Comprehensive Evaluation:**
```bash
python scripts/evaluate.py \
    --model_path experiments/checkpoints/best_model.pth \
    --data_path data/test \
    --output_dir results/evaluation
```

### **🎯 Key Features Implemented**

**Multi-Modal Training:**
- ✅ Supports staged and end-to-end training approaches
- ✅ Handles missing modalities gracefully  
- ✅ Multi-loss optimization (diagnosis + concept + consistency)

**Explainable AI:**
- ✅ Concept bottleneck layer with medical concept extraction
- ✅ HTML explanation reports with concept interpretations
- ✅ Visualization of concept predictions and feature importance

**Experiment Management:**
- ✅ Weights & Biases integration for experiment tracking
- ✅ Comprehensive checkpointing and model resumption
- ✅ Detailed evaluation reports with metrics and visualizations

**Production Ready:**
- ✅ Robust error handling and logging
- ✅ Configurable via YAML files
- ✅ Support for parallel data processing
- ✅ GPU/CPU automatic device detection

### **📊 System Architecture**

Your system now has the complete pipeline:

```
Raw Data → Preprocessing → Feature Extraction → Concept Bottleneck → CRF Classifier → Diagnosis + Explanations
```

The **concept bottleneck layer** is the key innovation, providing interpretable medical concepts that bridge the gap between raw features and final diagnosis, making the AI system transparent and trustworthy for medical applications.

### **🔬 Next Steps**

1. **Prepare data** in the expected CSV format with columns for `audio_path`, `eeg_path`, `text`, `diagnosis`
2. **Run preprocessing** to extract features from your multi-modal data
3. **Start training** with the end-to-end script using your configuration
4. **Evaluate results** and generate explanation reports for model validation






