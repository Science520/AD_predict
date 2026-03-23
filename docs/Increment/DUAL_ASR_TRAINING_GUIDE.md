# 双方案ASR微调指南

## 📋 方案概览

针对您的中文方言ASR任务（168个训练样本），我已经准备了两个训练方案：

| 方案 | 模型 | 状态 | 推荐度 | 说明 |
|------|------|------|--------|------|
| **方案1** | Whisper-medium | ⚠️ 需下载 | ⭐⭐⭐⭐ | 更适合小数据集 |
| **方案1-备选** | Whisper-large-v3 (优化配置) | ✅ 可用 | ⭐⭐⭐ | 使用已有模型 |
| **方案2** | FunASR Paraformer | 📝 框架已就绪 | ⭐⭐⭐⭐⭐ | 专为中文设计 |

---

## 🚀 方案1: Whisper微调

### 方案1A: Whisper-medium（推荐，但需下载）

**优势**：
- ✅ 参数量适中（769M），更适合168样本的数据集
- ✅ 对中文支持很好
- ✅ 训练速度更快，内存占用更少
- ✅ 过拟合风险较低

**配置文件**：`configs/training_args_medium.yaml`

**启动命令**：
```bash
# 1. 先下载模型（需要网络）
python scripts/download_whisper_medium.py

# 2. 启动训练
cd /home/saisai/AD_predict/AD_predict
conda activate graph
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=~/.cache/huggingface
export HF_ENDPOINT='https://hf-mirror.com'
export HF_HUB_OFFLINE=1

python scripts/2_finetune_whisper_lora.py \
    --config configs/training_args_medium.yaml \
    2>&1 | tee /tmp/whisper_medium_training.log
```

**⚠️ 当前问题**：网络无法访问Hugging Face，无法下载whisper-medium模型

**解决方案**：
1. 使用VPN或代理后重新运行 `python scripts/download_whisper_medium.py`
2. 或手动下载模型文件到 `~/.cache/huggingface/hub/`
3. **或使用方案1B（推荐）**

---

### 方案1B: Whisper-large-v3（优化配置，立即可用）✅

**优势**：
- ✅ **模型已在本地缓存，无需下载**
- ✅ 效果最好（如果不过拟合）
- ✅ 通过优化配置减少过拟合风险

**优化策略**：
1. **增大LoRA dropout**: 从0.05提高到0.1，增强正则化
2. **增大weight decay**: 从0.01提高到0.05
3. **减少训练轮数**: 从10减少到5-7轮，避免过拟合
4. **增加数据增强**: 确保增强数据正确加载

**创建优化配置**：

我为您创建一个优化的large-v3配置：


