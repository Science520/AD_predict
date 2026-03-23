# 🌙 今晚8小时实验计划

**当前时间**: 2025-10-22 下午  
**可用时间**: 8小时（今晚挂机）  
**目标**: 完成Whisper参数对比实验，为25号月报准备材料  

---

## ✅ **数据准备完成！**

```
✅ 数据增强成功
原始样本: 151个
增强样本: +604个
最终总数: 755个 (5倍增长!)

方言分布已优化，准备开始训练！
```

---

## 🚀 **一键启动（推荐）**

### 方案A: 只运行Whisper实验 ⭐⭐⭐⭐⭐ **强烈推荐**

```bash
cd /home/saisai/AD_predict/AD_predict

# 一键启动4组Whisper实验
nohup bash scripts/run_experiments.sh > /tmp/experiments.log 2>&1 &

# 查看进度
tail -f /tmp/experiments.log
```

**预计耗时**: 2-3小时  
**实验内容**:
1. ✅ 高LoRA rank (最有希望)
2. ✅ 低学习率 (稳定训练)
3. ✅ 大batch (提升稳定性)
4. ✅ 激进训练 (快速验证)

**为什么推荐**: 
- ✅ Whisper脚本已完全调试好
- ✅ 数据增强已完成(755样本)
- ✅ 失败自动继续下一个
- ✅ 明早100%有结果

---

### 方案B: Whisper + FunASR双模型实验 ⭐⭐⭐⭐ **如需FunASR对比**

```bash
cd /home/saisai/AD_predict/AD_predict

# 一键启动8组实验
nohup bash scripts/run_all_experiments.sh > /tmp/all_experiments.log 2>&1 &

# 查看进度
tail -f /tmp/all_experiments.log
```

**预计耗时**: 4-5小时  
**实验内容**:
- 4组Whisper实验
- 4组FunASR实验 (⚠️ FunASR脚本需要补充)

**注意**: FunASR部分可能需要调试，如果失败会自动跳过

---

## 📊 **实验配置对比**

| # | 实验名称 | LoRA rank | 学习率 | Epochs | Batch | 特点 |
|---|----------|-----------|--------|--------|-------|------|
| 1 | **exp1_high_rank** | 32 | 5e-5 | 25 | 1×16 | 更多参数 ⭐⭐⭐⭐⭐ |
| 2 | **exp2_low_lr** | 16 | 1e-5 | 30 | 1×16 | 稳定学习 ⭐⭐⭐⭐ |
| 3 | **exp3_large_batch** | 16 | 3e-5 | 20 | 2×8 | 大batch ⭐⭐⭐ |
| 4 | **exp4_aggressive** | 64 | 1e-4 | 15 | 1×16 | 激进训练 ⭐⭐ |

---

## 📈 **预期结果**

### 当前基准
```
Baseline (168样本):
- WER: 0.9984 (99.84%错误)
- 结论: 几乎完全失败
```

### 预期改进 (755样本)
```
乐观预期:
- WER: 0.5-0.7 (50-70%错误)  ✅ 可用
- 改进: 30-50%

保守预期:
- WER: 0.7-0.9 (70-90%错误)  ⚠️ 有改进
- 改进: 10-30%
```

---

## 🔍 **监控命令**

### 查看实时进度
```bash
# 实时日志
tail -f /tmp/experiments.log

# 快速检查状态
bash CHECK_STATUS.sh

# GPU使用情况
nvidia-smi

# 训练进度摘要
bash scripts/monitor_training.sh
```

### 查看中间结果
```bash
# 查看最新日志
ls -lt /tmp/whisper_logs/*.log | head -1 | xargs tail -50

# 查看已完成的模型
ls -lh /data/AD_predict/exp*/eval_results.json
```

---

## 📅 **明天早上（10月23日）**

### 步骤1: 查看实验结果 ⏱️ 5分钟

```bash
cd /home/saisai/AD_predict/AD_predict

# 找到实验结果目录
RESULTS_DIR=$(ls -td /data/AD_predict/experiments_* 2>/dev/null | head -1)

# 如果用的是all_experiments
RESULTS_DIR=$(ls -td /data/AD_predict/all_experiments_* 2>/dev/null | head -1)

# 查看最终报告
cat $RESULTS_DIR/FINAL_REPORT.md

# 查看摘要
cat $RESULTS_DIR/summary.txt
```

### 步骤2: 分析最优配置 ⏱️ 10分钟

```bash
# 生成详细分析
python scripts/analyze_experiments.py --results_dir $RESULTS_DIR

# 查看分析报告
cat $RESULTS_DIR/EXPERIMENT_REPORT.md
```

### 步骤3: 选择最优配置重新训练 ⏱️ 可选

```bash
# 假设exp1效果最好
python scripts/2_finetune_whisper_lora.py \
  --config configs/exp1_high_rank.yaml \
  > /tmp/final_best_training.log 2>&1 &
```

---

## 🎯 **25号交付物（已准备就绪）**

✅ **文档**:
- `TRAINING_REPORT.md` - 详细训练报告
- `ACTION_PLAN.md` - 行动计划
- `QUICK_START_GUIDE.md` - 快速指南

✅ **实验结果**:
- 4组参数对比（明早完成）
- WER改进对比
- 最优配置推荐

✅ **模型文件**:
- 训练好的LoRA适配器
- 可加载使用的模型

✅ **可视化**:
- TensorBoard训练曲线
- WER对比图表

---

## ⚠️ **如果遇到问题**

### 实验中断了？
```bash
# 检查进程
ps aux | grep "python.*finetune"

# 重新启动
nohup bash scripts/run_experiments.sh > /tmp/experiments_retry.log 2>&1 &
```

### GPU内存不足？
```bash
# 检查GPU
nvidia-smi

# 清理GPU
pkill -f "python.*finetune"
python3 -c "import torch; torch.cuda.empty_cache()"

# 使用最小配置
python scripts/2_finetune_whisper_lora.py \
  --config configs/training_args_medium_minimal.yaml
```

### 想停止实验？
```bash
# 杀掉所有训练进程
pkill -f "python.*finetune"

# 或者只杀掉某个
kill <PID>
```

---

## 💡 **睡前检查清单**

```bash
# 1. 启动实验
cd /home/saisai/AD_predict/AD_predict
nohup bash scripts/run_experiments.sh > /tmp/experiments.log 2>&1 &

# 2. 确认已启动
tail -20 /tmp/experiments.log

# 3. 检查GPU
nvidia-smi

# 4. 记下开始时间
date

# 5. 安心睡觉 😴
```

---

## 🎉 **明早预期**

您将看到：
1. ✅ 4个实验全部完成（或部分完成）
2. ✅ 完整的WER对比结果
3. ✅ 每个配置的训练日志
4. ✅ 最优配置推荐
5. ✅ 可视化训练曲线

**一切准备就绪！现在就启动实验吧！** 🚀

---

## 📞 **快速命令参考卡**

```bash
# 启动实验
cd /home/saisai/AD_predict/AD_predict && nohup bash scripts/run_experiments.sh > /tmp/experiments.log 2>&1 &

# 查看进度
tail -f /tmp/experiments.log

# 检查状态
bash CHECK_STATUS.sh

# GPU状态
nvidia-smi

# 停止实验
pkill -f "python.*finetune"
```

祝实验顺利！明早见！🌟

