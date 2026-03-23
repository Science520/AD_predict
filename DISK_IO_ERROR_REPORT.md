# 磁盘 I/O 错误诊断报告

**日期:** 2025-10-30  
**问题:** /data/AD_predict 目录下部分子目录出现 I/O 错误

## 问题症状

### 受影响的目录
以下目录无法读取内容（errno 5: Input/output error）：
- `/data/AD_predict/exp1_high_rank/` ✗
- `/data/AD_predict/exp2_low_lr/` ✗
- `/data/AD_predict/exp3_large_batch/` ✗
- `/data/AD_predict/exp4_aggressive/checkpoint-500/` ✗
- `/data/AD_predict/exp4_aggressive/checkpoint-645/` ✗
- `/data/AD_predict/exp4_aggressive/final_adapter/` ✗

### 正常可访问的目录
- `/data/AD_predict/data/` ✓
- `/data/AD_predict/experiments/` ✓
- `/home/saisai/AD_predict/AD_predict/models/best_model` ✓
- `/home/saisai/AD_predict/AD_predict/whisper_lora_dialect/` ✓

## 技术诊断

### 目录元数据正常
```bash
$ stat /data/AD_predict/exp1_high_rank
  File: /data/AD_predict/exp1_high_rank
  Size: 4096        Blocks: 8          IO Block: 4096   directory
  Device: fd00h/64768d  Inode: 78774963    Links: 6
  Access: (0775/drwxrwxr-x)  Uid: ( 1017/  saisai)   Gid: ( 1017/  saisai)
```
- inode 信息可读
- 权限正常 (775)
- 所有者正确

### 但目录内容无法读取
```bash
$ ls /data/AD_predict/exp1_high_rank/
ls: reading directory '.': Input/output error

$ python -c "import os; print(os.listdir('/data/AD_predict/exp1_high_rank'))"
OSError: [Errno 5] Input/output error
```

### 磁盘状态
```bash
$ df -h /data
Filesystem                   Size  Used Avail Use% Mounted on
/dev/mapper/vg_data-lv_data  7.3T  414G  6.5T   6% /data

$ mount | grep /data
/dev/mapper/vg_data-lv_data on /data type ext4 (rw,noatime)
```
- 磁盘未满（6% 使用率）
- 文件系统类型：ext4
- 挂载选项：rw,noatime

## 可能原因

1. **文件系统损坏**
   - ext4 文件系统的部分 inode 或 dentry 损坏
   - 可能由突然断电、系统崩溃或磁盘写入错误导致

2. **磁盘坏道**
   - 物理磁盘存在坏扇区
   - 需要检查 SMART 信息

3. **内核缓存问题**
   - dentry cache 损坏
   - 较少见，重启可能解决

## 建议解决方案

### 方案1: 文件系统检查（需要 root 权限）
```bash
# 注意：fsck 需要卸载文件系统或只读模式
sudo umount /data
sudo fsck.ext4 -f /dev/mapper/vg_data-lv_data
```

### 方案2: 检查磁盘健康
```bash
sudo smartctl -a /dev/mapper/vg_data-lv_data
sudo dmesg | grep -i "error\|disk"
```

### 方案3: 尝试恢复数据
```bash
# 如果有备份，直接使用备份
# 如果没有，可以尝试 debugfs 或 extundelete 工具
sudo debugfs /dev/mapper/vg_data-lv_data
```

## 当前应对策略

由于无法访问 exp1-4 目录中的训练模型，我们使用以下可用资源进行评估：

### 可用模型
1. **Whisper-Medium (原始基线)**
   - 路径：通过 HuggingFace 加载
   - 无需本地文件

2. **Best Model**
   - 路径：`/home/saisai/AD_predict/AD_predict/models/best_model`
   - 状态：✓ 可访问

3. **Dialect Final Adapter**
   - 路径：`/home/saisai/AD_predict/AD_predict/whisper_lora_dialect/final_adapter`
   - 状态：✓ 可访问

4. **Dialect Checkpoint-60**
   - 路径：`/home/saisai/AD_predict/AD_predict/whisper_lora_dialect/checkpoint-60`
   - 状态：✓ 可访问

### 评估脚本
创建了新的评估脚本，只使用本地项目目录下可访问的模型：
- `scripts/eval_seniortalk_available_models.py`
- `scripts/run_seniortalk_eval_tmux.sh`

## 后续行动

1. **紧急** - 联系系统管理员检查磁盘健康状态
2. **紧急** - 运行 fsck 修复文件系统
3. **重要** - 检查是否有训练模型的备份
4. **重要** - 如果数据无法恢复，考虑重新训练这些实验
5. **建议** - 建立定期备份机制
6. **建议** - 监控磁盘 SMART 信息

## 数据丢失评估

### 可能丢失的模型检查点
- Exp1 (高 Rank): checkpoint-100
- Exp2 (低学习率): checkpoint-1100  
- Exp3 (大 Batch): checkpoint-750
- Exp4 (激进学习率): checkpoint-500, checkpoint-645, final_adapter

### 仍可用的模型
- Best Model (最优模型) - 可能是 exp3 的某个checkpoint
- Dialect Final Adapter - 方言微调最终模型
- Dialect Checkpoint-60 - 方言微调中间检查点

## 结论

这是一个严重的文件系统问题，不是权限或配置问题。需要系统管理员介入进行磁盘修复。在修复之前，我们使用本地项目目录下的可用模型继续进行评估工作。


