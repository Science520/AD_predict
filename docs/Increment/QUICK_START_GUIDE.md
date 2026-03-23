# 🚀 Whisper方言ASR训练快速指南

**目标**: 25号前完成训练并生成对比报告  
**当前时间**: 2025-10-22  
**当前状态**: ✅ Baseline训练完成，WER=0.9984（效果差，需要优化）

---

## 📋 今天下午的行动清单

### ✅ 步骤1: 数据增强（立即执行）

```bash
cd /home/saisai/AD_predict/AD_predict

# 运行数据增强脚本
# 将168个样本扩展到500-700个
python scripts/1_prepare_dataset.py

# 预计时间: 10-15分钟
# 效果: 通过音频变换（速度、音调、噪声）生成更多样本
```

**预期输出**:
```
原始样本: 168
增强样本: ~500-700
保存位置: /data/AD_predict/data/raw/audio/elderly_audios_augmented/
```

---

### ✅ 步骤2: 验证增强后的数据

```bash
# 检查增强后的数据集大小
python scripts/0_validate_data.py

# 确认音频文件数量
ls /data/AD_predict/data/raw/audio/elderly_audios_augmented/*.wav | wc -l
```

---

### ✅ 步骤3: 重新预处理数据集

```bash
# 重新生成训练数据集（包含增强样本）
cd /home/saisai/AD_predict/AD_predict
python scripts/1_prepare_dataset.py
```

---

## 🌙 今晚挂机运行

### 方案A: 运行全部4个实验（推荐）⭐⭐⭐⭐⭐

```bash
cd /home/saisai/AD_predict/AD_predict

# 启动自动化实验
# 将依次运行4组参数配置
nohup bash scripts/run_experiments.sh > /tmp/experiments.log 2>&1 &

# 查看进度
tail -f /tmp/experiments.log
```

**预计耗时**: 2-3小时  
**实验内容**:
1. 实验1: 高LoRA rank (r=32)
2. 实验2: 低学习率 (lr=1e-5, 30 epochs)
3. 实验3: 大batch (batch=2×8)
4. 实验4: 激进训练 (r=64, lr=1e-4)

---

### 方案B: 只运行最有希望的2个（快速）⭐⭐⭐

```bash
cd /home/saisai/AD_predict/AD_predict

# 运行实验1
export CUDA_VISIBLE_DEVICES=1
conda activate graph
python scripts/2_finetune_whisper_lora.py --config configs/exp1_high_rank.yaml > /tmp/exp1.log 2>&1 &

# 等第一个完成后，运行实验2
# python scripts/2_finetune_whisper_lora.py --config configs/exp2_low_lr.yaml > /tmp/exp2.log 2>&1 &
```

**预计耗时**: 1-1.5小时

---

## 📊 明天（10月23日）分析结果

### 步骤1: 生成实验对比报告

```bash
cd /home/saisai/AD_predict/AD_predict

# 找到实验结果目录
RESULTS_DIR=$(ls -td /data/AD_predict/experiments_* | head -1)

# 生成分析报告
python scripts/analyze_experiments.py --results_dir $RESULTS_DIR

# 查看报告
cat $RESULTS_DIR/EXPERIMENT_REPORT.md
```

### 步骤2: 选择最优配置并训练完整模型

```bash
# 假设实验2效果最好
# 用最优配置训练更多epoch
python scripts/2_finetune_whisper_lora.py \
  --config configs/exp2_low_lr.yaml \
  > /tmp/final_training.log 2>&1 &
```

### 步骤3: 生成对比报告（给月报用）

```bash
python scripts/generate_comparison_report.py \
  --baseline /data/AD_predict/whisper_medium_minimal \
  --best_model /data/AD_predict/exp2_low_lr \
  --output monthly_report.md
```

---

## 📈 预期改进效果

### 当前状态
- **Baseline WER**: 0.9984 (99.84%错误率) ❌
- **数据量**: 168样本

### 预期改进（数据增强后）
- **目标WER**: 0.6-0.8 (60-80%错误率) ✅
- **数据量**: 500-700样本
- **改进幅度**: 20-40%

### 关键成功因素
1. ✅ 数据增强效果好（增加3-4倍样本）
2. ✅ 参数调优找到最优配置
3. ⚠️ 方言分布更均衡（部分方言仍可能不足）

---

## 🆘 如果效果仍然不好

### Plan B: 切换到中文专用模型

```bash
# 1. 尝试FunASR（阿里达摩院，专注中文）
python scripts/2_finetune_funasr_lora_FunASR.py

# 2. 或者尝试更小的Whisper模型
# whisper-small (244M参数，更适合小数据集)
```

---

## 📝 实验参数对比表

| 实验 | LoRA rank | 学习率 | Epochs | Batch | 特点 |
|------|-----------|--------|--------|-------|------|
| **Baseline** | 8 | 3e-5 | 20 | 1×16 | 当前配置 |
| **实验1** | 32 | 5e-5 | 25 | 1×16 | 更多可训练参数 |
| **实验2** | 16 | 1e-5 | 30 | 1×16 | 稳定学习 |
| **实验3** | 16 | 3e-5 | 20 | 2×8 | 大batch |
| **实验4** | 64 | 1e-4 | 15 | 1×16 | 激进训练 |

---

## 🎯 25号交付物

1. ✅ **训练报告** (TRAINING_REPORT.md)
2. ✅ **实验对比** (EXPERIMENT_REPORT.md)
3. ✅ **最优模型** (/data/AD_predict/best_model/)
4. ✅ **对比结果** (微调前 vs 微调后的WER)
5. ✅ **TensorBoard可视化** (训练曲线)

---

## 💡 Tips

### 监控训练进度
```bash
# 实时查看训练日志
tail -f /tmp/exp1.log

# 查看GPU使用
nvidia-smi

# 查看训练进度
bash scripts/monitor_training.sh
```

### TensorBoard可视化
```bash
# 启动TensorBoard
tensorboard --logdir=/data/AD_predict/logs_exp1 --port 6006

# 在浏览器访问
http://localhost:6006
```

---

## ❓ 常见问题

### Q: 训练中断了怎么办？
```bash
# 从最新checkpoint恢复训练
python scripts/2_finetune_whisper_lora.py \
  --config configs/exp1_high_rank.yaml \
  --resume_from_checkpoint auto
```

### Q: GPU内存不足？
```bash
# 使用更小的配置
python scripts/2_finetune_whisper_lora.py \
  --config configs/training_args_medium_minimal.yaml
```

### Q: 想要更快看到结果？
```bash
# 减少epoch数，快速验证
# 修改配置文件中的 num_train_epochs: 10
```

---

## 📞 需要帮助？

查看详细文档:
- `TRAINING_REPORT.md` - 训练结果分析
- `DUAL_ASR_TRAINING_GUIDE.md` - 双模型训练指南
- `TRAINING_GUIDE.md` - 完整训练指南

祝训练顺利！🎉


