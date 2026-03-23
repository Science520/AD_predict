# 模型评估任务 - 最终信息

**启动时间:** 2025-10-30 12:27:56  
**虚拟环境:** graph (Python 3.13.2)  
**会话名:** seniortalk_eval  
**状态:** ✅ 正在运行

---

## 评估模型列表（8个）

### Whisper 系列（7个）

| # | 模型名称 | 基础模型 | 说明 |
|---|---------|----------|------|
| 1 | Whisper-Medium 原始 | whisper-medium | 未微调基线 |
| 2 | Exp1: High Rank | whisper-medium | LoRA r=32, alpha=64 |
| 3 | Exp2: Low LR | whisper-medium | 低学习率 checkpoint-1100 |
| 4 | Exp3: Large Batch | whisper-medium | 大批次 checkpoint-750 |
| 5 | Exp4: Aggressive LR | whisper-medium | 激进学习率 checkpoint-500 |
| 6 | **Best Model** | whisper-medium | **最优保存模型** |
| 7 | **Dialect Final** | **whisper-large-v3** | **方言微调（Large模型）** ⭐ |

### FunASR（可选单独测试）

| # | 模型名称 | 来源 | 状态 |
|---|---------|------|------|
| 8 | FunASR SenseVoice | 阿里云 iic/SenseVoiceSmall | 脚本已准备 |

---

## 关键模型说明

### 🌟 Dialect Final (whisper-large-v3)
- **基础:** Whisper-Large-v3（比 Medium 大很多，性能更强）
- **训练:** LoRA 微调，r=16, alpha=32
- **目标模块:** q_proj, v_proj
- **用途:** 专门针对方言优化

### 💎 Best Model
- **基础:** Whisper-Medium
- **来源:** 从 Exp1-4 中选出的最优模型
- **位置:** `/home/saisai/AD_predict/AD_predict/models/best_model`

---

## 测试配置

### 数据集
- **来源:** SeniorTalk sentence_data test set
- **样本数:** 100个（均匀采样）
- **特点:** 老年人真实语音，包含方言

### 评估指标
1. **WER** - 词错误率（越低越好）
2. **CER** - 字错误率（越低越好）
3. **字准确率** - 字符匹配准确度（越高越好）
4. **音调准确率** - 在字正确时计算音调（越高越好）

---

## 查看实时进度

### 方法1: 附加到 tmux 会话
```bash
tmux attach -t seniortalk_eval
# 按 Ctrl+B 然后按 D 分离
```

### 方法2: 查看日志
```bash
tail -f /data/AD_predict/experiments/logs/seniortalk_eval_20251030_122756.log
```

### 方法3: 快速检查
```bash
tmux capture-pane -t seniortalk_eval -p | tail -30
```

---

## 单独运行 FunASR（可选）

如果想测试阿里云的 FunASR 模型：

```bash
# 在 graph 环境中运行
conda activate graph
cd /home/saisai/AD_predict/AD_predict
python scripts/eval_funasr_seniortalk.py 2>&1 | tee /data/AD_predict/experiments/logs/funasr_eval.log
```

---

## 预计完成时间

### Whisper 评估
- 每个模型: 10-15 分钟
- 7个模型总计: **70-105 分钟**
- **预计完成时间:** 约凌晨 2:00

### 注意
- Dialect Final (Large-v3) 可能需要更长时间（模型更大）
- GPU 使用率会很高，正常现象

---

## 结果文件位置

完成后查看：

```
/data/AD_predict/experiments/seniortalk_eval_available/
├── evaluation_summary.json          # 📊 总体汇总
├── comparison_table.csv             # 📈 CSV对比表
├── comparison_table.md              # 📝 Markdown报告
└── [model_name]_results.json        # 🔍 每个模型详细结果
```

---

## 技术细节

### 磁盘问题解决方案
- ✅ **问题:** `/data/AD_predict/exp*` 目录列表损坏
- ✅ **解决:** 直接用完整路径访问文件，成功绕过
- ✅ **结果:** 所有模型文件完好，可正常加载

### 虚拟环境
- ✅ **使用:** graph (Python 3.13.2)
- ✅ **原因:** 避免 peft/transformers 版本冲突
- ✅ **状态:** 已正确激活并运行

---

## 快速命令

### 停止评估
```bash
tmux kill-session -t seniortalk_eval
```

### 查看GPU
```bash
nvidia-smi
watch -n 1 nvidia-smi  # 持续监控
```

### 检查完成状态
```bash
ls -lh /data/AD_predict/experiments/seniortalk_eval_available/*.json
```

---

## 总结

✅ **已解决:** 磁盘 I/O 问题（文件都在，用路径访问）  
✅ **已解决:** 依赖冲突（使用 graph 虚拟环境）  
✅ **正在运行:** 7个 Whisper 模型评估  
📊 **测试中:** 100个 SeniorTalk 老年人语音样本  
🎯 **目标:** 找出最适合老年人方言的 ASR 模型  

🌙 **建议:** 现在可以安心休息，明早查看结果！

---

## 额外说明

### Whisper-Large-v3 vs Medium
- **Large-v3:** 1550M 参数，性能最强
- **Medium:** 769M 参数，速度较快
- **期待:** Large-v3 的 Dialect Final 可能表现最好

### 关于 Best Model
这个模型最可能是 **Exp3: Large Batch** 的某个 checkpoint，因为：
- Large Batch 训练通常更稳定
- checkpoint-750 是训练中期，可能是最优点
- 被单独保存为 "best_model" 说明验证集表现最好

明早见！🚀


