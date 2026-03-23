# 模型评估状态

**启动时间:** 2025-10-30 12:24:52  
**会话名:** seniortalk_eval

---

## 正在评估的模型（7个）

### Whisper 系列（6个 + 1个Large）

| # | 模型 | 基础模型 | 说明 |
|---|------|----------|------|
| 1 | Whisper-Medium 原始 | whisper-medium | 未微调基线 |
| 2 | Exp1: High Rank | whisper-medium | LoRA rank=32, alpha=64 |
| 3 | Exp2: Low LR | whisper-medium | 低学习率训练 |
| 4 | Exp3: Large Batch | whisper-medium | 大批次训练 |
| 5 | Exp4: Aggressive LR | whisper-medium | 激进学习率 |
| 6 | **Best Model** | whisper-medium | **保存的最优模型** |
| 7 | **Dialect Final** | **whisper-large-v3** | **方言微调（Large模型）** |

### FunASR（单独评估）

**模型:** FunASR SenseVoice (iic/SenseVoiceSmall)  
**状态:** 脚本已准备，可单独运行

---

## 评估配置

### 测试数据
- **数据集:** SeniorTalk sentence_data test set
- **样本数:** 100个（从test set均匀采样）
- **数据源:** parquet files

### 评估指标
1. **WER** (Word Error Rate) - 词错误率
2. **CER** (Character Error Rate) - 字错误率  
3. **字准确率** (Character Accuracy)
4. **音调准确率** (Tone Accuracy) - 仅在字正确时计算

---

## 查看进度

### 方法1: 附加到 tmux 会话
```bash
tmux attach -t seniortalk_eval
# 分离: Ctrl+B 然后按 D
```

### 方法2: 查看实时日志
```bash
tail -f /data/AD_predict/experiments/logs/seniortalk_eval_20251030_122452.log
```

### 方法3: 列出所有会话
```bash
tmux ls
```

---

## 模型说明

### Best Model vs Exp1-4
**Best Model** 是从 exp1-4 中选出的最优模型，可能是：
- 基于验证集性能选择
- 基于 WER/CER 指标选择
- 保存在 `/home/saisai/AD_predict/AD_predict/models/best_model`

### Dialect Final
- **基础模型:** whisper-large-v3 ✨（比 medium 更大更强）
- **训练方式:** LoRA 微调
- **目标:** 方言识别优化
- **配置:** r=16, alpha=32, target_modules=[q_proj, v_proj]

---

## 单独运行 FunASR 评估

如果想要评估 FunASR（阿里云语音识别）：

```bash
cd /home/saisai/AD_predict/AD_predict
python scripts/eval_funasr_seniortalk.py 2>&1 | tee /data/AD_predict/experiments/logs/funasr_eval.log
```

**注意:** FunASR 会自动从 ModelScope 下载模型，首次运行需要时间。

---

## 预期完成时间

### Whisper 模型评估
- 每个模型: ~10-15分钟
- 7个模型总计: **70-105分钟** (1-2小时)
- **预计完成:** 约 2:00 AM 左右

### FunASR 评估（如果运行）
- 单个模型: ~10-15分钟

---

## 输出文件

评估完成后会生成：

### Whisper 评估结果
```
/data/AD_predict/experiments/seniortalk_eval_available/
├── evaluation_summary.json          # 总体汇总
├── comparison_table.csv             # CSV格式对比表
├── comparison_table.md              # Markdown格式报告
├── whisper_medium_baseline_results.json
├── exp1_high_rank_results.json
├── exp2_low_lr_results.json
├── exp3_large_batch_results.json
├── exp4_aggressive_results.json
├── best_model_results.json
└── dialect_final_results.json
```

### FunASR 评估结果（如果运行）
```
/data/AD_predict/experiments/funasr_eval/
└── funasr_results.json
```

---

## 快速命令

### 停止评估
```bash
tmux kill-session -t seniortalk_eval
```

### 查看GPU使用
```bash
nvidia-smi
watch -n 1 nvidia-smi  # 持续监控
```

### 检查是否完成
```bash
ls -lh /data/AD_predict/experiments/seniortalk_eval_available/*.json
```

---

## 总结

✅ **已启动:** 7个Whisper模型评估（包括1个Large-v3）  
⏳ **进行中:** 在tmux后台运行  
📝 **日志:** `/data/AD_predict/experiments/logs/seniortalk_eval_20251030_122452.log`  
🎯 **目标:** 对比所有模型在老年人语音上的表现  

🌙 **建议:** 现在可以休息，明早查看结果！


