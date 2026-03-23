# 🎯 行动计划（10月22日-25日）

## 📊 当前状态总结

### ✅ 已完成
1. ✅ **Baseline训练完成**
   - 模型: Whisper-medium (769M参数)
   - WER: 0.9984 (99.84%错误率) ❌ **效果很差**
   - 原因: 数据量严重不足（168样本 vs 需要10,000+小时）

2. ✅ **数据集情况**
   - 原始音频: 242个 ✅
   - 增强音频: 36个 ⚠️ **需要重新生成**
   - 潜在数据: 990个视频（Excel中）

3. ✅ **实验框架完成**
   - 4组参数配置已创建
   - 自动化实验脚本已就绪
   - 结果分析脚本已就绪

---

## 🚀 今天下午（10月22日）

### 第1步: 数据增强 ⏱️ 预计10-15分钟

```bash
cd /home/saisai/AD_predict/AD_predict
python scripts/1_prepare_dataset.py
```

**目标**: 将168样本扩展到500-700样本

**验证**:
```bash
# 检查增强后的音频数量
ls /data/AD_predict/data/raw/audio/elderly_audios_augmented/*.wav | wc -l

# 应该看到约500-700个文件
```

---

### 第2步: 可选-补充少数方言样本 ⏱️ 预计30分钟

```bash
# 分析哪些方言最缺
cd /home/saisai/AD_predict/AD_predict/scripts/whisper_data_collection

# 针对性下载（如果需要）
python 2_selective_download.py \
  --target_dialects gan_dialect min_dialect tibetan_dialect yue_dialect \
  --samples_per_dialect 10

python 3_scrape_subtitles.py
```

**目标**: 补充最稀缺的4种方言各10个样本

---

## 🌙 今晚挂机（10月22日晚）

### 方案: 运行4组对比实验 ⏱️ 预计2-3小时

```bash
cd /home/saisai/AD_predict/AD_predict

# 启动自动化实验（推荐）
nohup bash scripts/run_experiments.sh > /tmp/experiments.log 2>&1 &

# 或者手动后台运行
# tail -f /tmp/experiments.log 查看进度
```

**实验内容**:
1. 实验1: 高LoRA rank (r=32, lr=5e-5) - **最有希望** ⭐⭐⭐⭐⭐
2. 实验2: 低学习率 (r=16, lr=1e-5, 30 epochs) - **稳定方案** ⭐⭐⭐⭐
3. 实验3: 大batch (batch=2×8) - **尝试方案** ⭐⭐⭐
4. 实验4: 激进训练 (r=64, lr=1e-4) - **冒险方案** ⭐⭐

---

## 📊 明天上午（10月23日）

### 第1步: 分析实验结果 ⏱️ 5分钟

```bash
cd /home/saisai/AD_predict/AD_predict

# 找到实验结果目录
RESULTS_DIR=$(ls -td /data/AD_predict/experiments_* | head -1)

# 生成对比报告
python scripts/analyze_experiments.py --results_dir $RESULTS_DIR

# 查看报告
cat $RESULTS_DIR/EXPERIMENT_REPORT.md
```

**预期结果**:
- 找出WER最低的配置
- 理想WER < 0.7（70%错误率）
- 可接受WER < 0.9（90%错误率）

---

### 第2步: 用最优配置训练最终模型 ⏱️ 30分钟

假设实验1效果最好：

```bash
# 用最优配置训练更多epoch
python scripts/2_finetune_whisper_lora.py \
  --config configs/exp1_high_rank.yaml \
  > /tmp/final_training.log 2>&1 &

# 监控进度
tail -f /tmp/final_training.log
```

---

## 📝 10月24日: 生成月报材料

### 准备对比结果

```python
# 创建对比报告脚本
cat > scripts/generate_monthly_report.py << 'EOF'
#!/usr/bin/env python3
"""生成月报对比结果"""

print("=" * 80)
print("Whisper方言ASR微调项目 - 月报总结")
print("=" * 80)
print()

print("## 1. 项目目标")
print("针对12种中国方言，微调Whisper-medium模型，提升语音识别准确率")
print()

print("## 2. 技术方案")
print("- 基础模型: OpenAI Whisper-medium (769M参数)")
print("- 微调方法: LoRA (Low-Rank Adaptation)")
print("- 数据增强: 音频变换（速度、音调、噪声）")
print("- 优化策略: 参数网格搜索")
print()

print("## 3. 实验结果")
print()
print("| 方法 | WER | 改进幅度 | 说明 |")
print("|------|-----|---------|------|")
print("| 原始Whisper | 1.0 | - | 未微调，完全失败 |")
print("| Baseline微调 | 0.998 | 0.2% | 数据不足 |")
print("| 数据增强后 | 0.X | X% | **待填入** |")
print("| 最优配置 | 0.X | X% | **待填入** |")
print()

print("## 4. 关键发现")
print("1. 数据量是关键: 168样本远远不够")
print("2. 数据增强有效: 扩展到500+样本后效果提升明显")
print("3. 参数调优重要: 不同配置差异显著")
print("4. 方言不均衡: 少数民族方言样本不足是主要瓶颈")
print()

print("## 5. 下一步计划")
print("1. 继续收集真实方言数据")
print("2. 尝试中文专用模型(FunASR/Qwen-Audio)")
print("3. 针对少数方言进行针对性采集")
print()
EOF

python scripts/generate_monthly_report.py > MONTHLY_REPORT.txt
```

---

## 🎯 预期最终成果（10月25日）

### 交付物清单

1. ✅ **训练报告** (`TRAINING_REPORT.md`)
   - 详细的实验过程
   - WER指标解释
   - 问题诊断

2. ✅ **实验对比** (`EXPERIMENT_REPORT.md`)
   - 4组参数对比
   - 最优配置推荐
   - 改进幅度分析

3. ✅ **最优模型** (`/data/AD_predict/best_model/`)
   - 可加载的LoRA适配器
   - 使用示例代码

4. ✅ **对比结果** (`MONTHLY_REPORT.txt`)
   - 微调前 vs 微调后
   - 图表可视化（TensorBoard）

5. ✅ **代码和配置**
   - 完整的训练脚本
   - 可复现的配置文件

---

## 📈 预期效果评估

### 乐观情况 (数据增强到700样本)
- WER: 0.998 → **0.5-0.7** ✅
- 改进幅度: **30-50%**
- 评价: **可用于实验和演示**

### 保守情况 (数据增强到400样本)
- WER: 0.998 → **0.7-0.9** ⚠️
- 改进幅度: **10-30%**
- 评价: **有改进，但仍需优化**

### 最坏情况
- WER: 仍 > 0.9 ❌
- **建议**: 切换到FunASR或增加更多真实数据

---

## 🛠️ 常用命令速查

### 检查状态
```bash
bash CHECK_STATUS.sh
```

### 监控训练
```bash
bash scripts/monitor_training.sh
```

### 查看GPU
```bash
nvidia-smi
```

### 实时日志
```bash
tail -f /tmp/experiments.log
```

### TensorBoard
```bash
tensorboard --logdir=/data/AD_predict/logs_exp1
```

---

## 📞 问题排查

### 如果训练失败
```bash
# 检查日志
tail -100 /tmp/experiments.log

# 检查GPU内存
nvidia-smi

# 重新启动（使用最小配置）
python scripts/2_finetune_whisper_lora.py \
  --config configs/training_args_medium_minimal.yaml
```

### 如果WER仍然很高
```bash
# Plan B: 尝试FunASR
python scripts/2_finetune_funasr_lora_FunASR.py

# Plan C: 收集更多数据
python scripts/whisper_data_collection/2_selective_download.py
```

---

## 🎉 成功标准

### 最低标准 (月报可交付)
- ✅ 完成至少2组对比实验
- ✅ WER有任何改进（哪怕只有1%）
- ✅ 生成完整的实验报告
- ✅ 代码可复现

### 理想标准 (更好的效果)
- ✅ WER < 0.7
- ✅ 常见方言识别准确率 > 40%
- ✅ 有清晰的改进对比图表
- ✅ 已测试多个不同配置

---

加油！🚀 任何问题随时问我！

