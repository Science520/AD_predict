# 磁盘状态和模型可用性报告

**日期:** 2025-10-30  
**问题:** 部分 `/data/AD_predict` 目录有 I/O 错误

---

## 磁盘问题诊断

### 受影响的目录
以下目录无法读取（errno 5: Input/output error）：
- `/data/AD_predict/exp1_high_rank/` ✗
- `/data/AD_predict/exp2_low_lr/` ✗  
- `/data/AD_predict/exp3_large_batch/` ✗
- `/data/AD_predict/exp4_aggressive/` 的子目录 ✗
- 部分 SeniorTalk parquet 文件 ✗

### 可正常访问的资源
- `/data/AD_predict/data/raw/audio/` ✓
- `/data/AD_predict/data/raw/seniortalk_full/sentence_data/test-00000-of-00003.parquet` ✓ (第一个test文件可用，1956样本)
- `/data/AD_predict/data/raw/seniortalk_full/sentence_data/test-00001-of-00003.parquet` ✓ (1956样本)
- `/home/saisai/AD_predict/AD_predict/models/best_model` ✓
- `/home/saisai/AD_predict/AD_predict/whisper_lora_dialect/final_adapter` ✓
- `/home/saisai/AD_predict/AD_predict/whisper_lora_dialect/checkpoint-60` ✓

## 可用模型

由于磁盘 I/O 错误，只有以下模型可用于评估：

| 模型 | 路径 | 状态 |
|------|------|------|
| Whisper-Medium (原始) | HuggingFace加载 | ✓ 可用 |
| Best Model | `/home/saisai/AD_predict/AD_predict/models/best_model` | ✓ 可用 |
| Dialect Final | `/home/saisai/AD_predict/AD_predict/whisper_lora_dialect/final_adapter` | ✓ 可用 |
| Dialect Checkpoint-60 | `/home/saisai/AD_predict/AD_predict/whisper_lora_dialect/checkpoint-60` | ✓ 可用 |
| Exp1 High Rank | `/data/AD_predict/exp1_high_rank/checkpoint-100` | ✗ I/O错误 |
| Exp2 Low LR | `/data/AD_predict/exp2_low_lr/checkpoint-1100` | ✗ I/O错误 |
| Exp3 Large Batch | `/data/AD_predict/exp3_large_batch/checkpoint-750` | ✗ I/O错误 |
| Exp4 Aggressive | `/data/AD_predict/exp4_aggressive/checkpoint-*` | ✗ I/O错误 |

## 测试数据

### SeniorTalk 测试集
- 可用的 parquet 文件：2个（共3个）
- 可用样本总数：约 3,900+ 个
- 足够用于评估

## 建议行动

### 立即可做的事
1. **评估现有可用模型** - 使用 4 个可访问的模型进行评估
   - 已创建评估脚本，使用本地可用的模型
   - 测试数据：SeniorTalk test set (前2个parquet文件，共约3900个样本)

2. **记录当前评估结果** - 至少对比 4 个模型：
   - Whisper-Medium 原始
   - Best Model  
   - Dialect Final
   - Dialect Checkpoint-60

### 需要系统管理员协助
3. **修复文件系统** - 需要 root 权限
   ```bash
   # 检查磁盘健康
   sudo smartctl -a /dev/mapper/vg_data-lv_data
   
   # 文件系统检查（需要卸载或只读模式）
   sudo umount /data
   sudo fsck.ext4 -f /dev/mapper/vg_data-lv_data
   ```

4. **查看系统日志**
   ```bash
   sudo dmesg | tail -100 | grep -i "error\|i/o"
   sudo journalctl -xe | grep -i "data"
   ```

### 后续建议
5. **数据备份策略** - 防止将来数据丢失
   - 定期备份训练好的模型
   - 使用 rsync 或 tar 定期备份到其他存储

6. **如果无法修复** - 重新训练的替代方案
   - Best Model 可能已经是最优模型
   - 可以使用 SeniorTalk 数据重新训练新模型

## 当前评估计划

鉴于磁盘问题，调整评估计划为：

### 评估指标（保持不变）
1. WER (词错误率)
2. CER (字错误率)
3. 字准确率
4. 音调准确率（字正确时）

### 评估模型（调整为4个）
1. Whisper-Medium (原始基线)
2. Best Model
3. Dialect Final Adapter
4. Dialect Checkpoint-60

### 评估数据
- 数据集：SeniorTalk sentence_data test set  
- 样本数：100个（从可用的~3900个中采样）
- 数据来源：前2个可用的 parquet 文件

### 运行方式
- Tmux 后台运行
- 日志文件：`/data/AD_predict/experiments/logs/seniortalk_eval_*.log`
- 结果保存：`/data/AD_predict/experiments/seniortalk_eval_quick/`

## 总结

**好消息：**
- 有4个模型可用于评估
- SeniorTalk测试数据大部分可访问（约3900个样本）
- 足够完成模型对比评估任务

**需要注意：**
- `/data/AD_predict/exp*` 目录有文件系统问题
- 需要管理员权限修复
- 建议尽快修复或备份其他重要数据

**现在可以做：**
- 启动评估脚本，对比4个可用模型
- 生成评估报告和对比表格
- 在后台运行，明早查看结果


