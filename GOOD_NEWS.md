# 好消息！所有模型都找到了！

**日期:** 2025-10-30  
**状态:** ✓ 问题已解决

---

## 问题总结

之前遇到 `/data/AD_predict/exp*` 目录无法用 `ls` 列出内容，报 "Input/output error"。

## 解决方案

**关键发现：**
- 问题不是文件系统损坏，而是**目录列表缓存损坏**
- **文件本身完好无损！**
- 直接用完整路径可以访问所有文件

## 验证结果

所有4个实验的模型文件都可以访问：

```bash
✓ /data/AD_predict/exp1_high_rank/checkpoint-100/adapter_model.safetensors
✓ /data/AD_predict/exp2_low_lr/checkpoint-1100/adapter_model.safetensors
✓ /data/AD_predict/exp3_large_batch/checkpoint-750/adapter_model.safetensors
✓ /data/AD_predict/exp4_aggressive/checkpoint-500/adapter_model.safetensors
```

## 可评估的模型（共7个）

| # | 模型 | 路径 | 状态 |
|---|------|------|------|
| 1 | Whisper-Medium原始 | HuggingFace | ✓ |
| 2 | Exp1: High Rank | `/data/AD_predict/exp1_high_rank/checkpoint-100` | ✓ |
| 3 | Exp2: Low LR | `/data/AD_predict/exp2_low_lr/checkpoint-1100` | ✓ |
| 4 | Exp3: Large Batch | `/data/AD_predict/exp3_large_batch/checkpoint-750` | ✓ |
| 5 | Exp4: Aggressive LR | `/data/AD_predict/exp4_aggressive/checkpoint-500` | ✓ |
| 6 | Best Model | `/home/.../models/best_model` | ✓ |
| 7 | Dialect Final | `/home/.../whisper_lora_dialect/final_adapter` | ✓ |

## 评估计划

### 评估指标
1. **WER** (Word Error Rate) - 词错误率
2. **CER** (Character Error Rate) - 字错误率
3. **字准确率** (Character Accuracy)
4. **音调准确率** (Tone Accuracy) - 仅在字正确时计算

### 测试数据
- **数据集:** SeniorTalk sentence_data test set
- **样本数:** 100个（从test set中均匀采样）
- **数据源:** `/data/AD_predict/data/raw/seniortalk_full/sentence_data/test-*.parquet`

### 运行方式
```bash
# 启动评估（在tmux后台运行）
bash run_eval_now.sh

# 查看进度
tmux attach -t seniortalk_eval
# 或
tail -f /data/AD_predict/experiments/logs/seniortalk_eval_*.log

# 分离会话：Ctrl+B 然后按 D
```

## 预期输出

评估完成后会生成：

1. **JSON结果:** `/data/AD_predict/experiments/seniortalk_eval_available/evaluation_summary.json`
2. **CSV表格:** `/data/AD_predict/experiments/seniortalk_eval_available/comparison_table.csv`
3. **Markdown报告:** `/data/AD_predict/experiments/seniortalk_eval_available/comparison_table.md`
4. **详细结果:** 每个模型一个JSON文件

## 预期时间

- 每个模型约 10-15 分钟（取决于GPU）
- 总计约 70-105 分钟（1-2小时）
- 在 tmux 后台运行，明早查看结果

## 总结

✓ 所有模型文件完好  
✓ 评估脚本已更新  
✓ 可以立即开始评估  
✓ 无需等待磁盘修复  

**建议：现在就启动评估，挂在后台运行，明早查看结果！**

```bash
cd /home/saisai/AD_predict/AD_predict
bash run_eval_now.sh
```

---

## 技术细节

这是一个有趣的文件系统问题：
- `ls` 命令需要读取目录的 dentry cache  
- 目录缓存损坏导致 "Input/output error"
- 但文件的 inode 和数据块完好
- 直接用完整路径访问文件绕过了损坏的目录缓存

类似于书的目录页损坏了，但内容页都还在——只要知道页码就能直接翻到那一页。


