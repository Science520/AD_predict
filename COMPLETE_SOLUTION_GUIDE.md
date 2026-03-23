# 完整解决方案指南

## 📋 问题总结

你在运行 ASR 评估脚本时遇到了两个主要问题：

1. **权限问题**：无法写入 `/data/cache/hf/datasets` 目录
2. **只读文件系统**：`/data` 分区变成只读模式

## ✅ 解决方案概览

我为你准备了两套解决方案：

### 方案 A：使用用户主目录（立即可用）✨

**优点**：
- ✓ 无需修复 /data，立即可用
- ✓ 所有数据在你的主目录，权限无问题
- ✓ 更安全，不会影响其他用户

**缺点**：
- ⚠ 会占用主目录磁盘空间
- ⚠ 需要重新下载 Hugging Face 模型到新缓存位置

### 方案 B：修复 /data 分区（推荐长期使用）

**优点**：
- ✓ 恢复原有工作环境
- ✓ 可以利用 /data 的大容量空间

**缺点**：
- ⚠ 需要 sudo 权限
- ⚠ 修复时会暂时中断 /data 访问

---

## 🚀 方案 A：使用用户主目录（推荐先试这个）

### 1. 新文件位置说明

我已经创建了修改后的脚本，所有路径如下：

```
📁 用户主目录结构
/home/saisai/
├── .cache/huggingface/          # HF 模型缓存（自动创建）
│   ├── transformers/
│   ├── datasets/
│   └── hub/
├── AD_predict_results/          # 评估结果输出
│   └── seniortalk_evaluation/
│       ├── evaluation_summary.json
│       ├── comparison_table.csv
│       ├── comparison_table.md
│       └── *_results.json
└── AD_predict_logs/             # 运行日志
    └── eval_seniortalk_*.log
```

### 2. 运行评估

```bash
# 进入项目目录
cd ~/AD_predict/AD_predict

# 启动评估（在 tmux 中后台运行）
./run_eval_fixed.sh

# 查看运行状态（按 Ctrl+B 然后按 D 退出）
tmux attach -t seniortalk_eval_fixed

# 实时查看日志
tail -f ~/AD_predict_logs/eval_seniortalk_*.log
```

### 3. 查看结果

```bash
# 查看所有结果文件
ls -lh ~/AD_predict_results/seniortalk_evaluation/

# 查看汇总表格
cat ~/AD_predict_results/seniortalk_evaluation/comparison_table.md

# 查看详细 JSON
cat ~/AD_predict_results/seniortalk_evaluation/evaluation_summary.json
```

### 4. 停止评估（如需要）

```bash
# 杀掉 tmux 会话
tmux kill-session -t seniortalk_eval_fixed
```

---

## 🔧 方案 B：修复 /data 分区

### 快速修复（推荐）

```bash
# 运行自动修复脚本（需要输入你的密码）
cd ~/AD_predict/AD_predict
sudo ./fix_data_partition.sh
```

脚本会自动：
1. 诊断问题
2. 尝试简单修复（重新挂载）
3. 如果需要，进行深度修复（卸载、fsck、重新挂载）
4. 验证修复结果

### 手动修复步骤

如果自动脚本失败，可以手动操作：

#### 步骤 1：简单重新挂载

```bash
# 重新挂载为读写模式
sudo mount -o remount,rw /data

# 测试写入
touch /data/test_$(date +%s).txt && echo "✓ 修复成功" || echo "✗ 仍然只读"
```

#### 步骤 2：如果步骤 1 失败，进行深度修复

```bash
# 1. 检查谁在使用 /data
sudo lsof +D /data | head -n 20

# 2. 卸载（如果有 Docker，先停止）
sudo umount /data

# 3. 文件系统检查并修复
sudo e2fsck -f -y /dev/mapper/vg_data-lv_data

# 4. 重新挂载
sudo mount /data

# 5. 验证
df -h /data
touch /data/test_$(date +%s).txt
```

### 诊断命令

如果修复失败，运行完整诊断：

```bash
sudo bash << 'EOF'
echo "=== 文件系统状态 ==="
tune2fs -l /dev/mapper/vg_data-lv_data | grep -i state

echo ""
echo "=== 挂载状态 ==="
mount | grep /data

echo ""
echo "=== 系统日志错误 ==="
dmesg | grep -iE "/data|error|readonly" | tail -n 20

echo ""
echo "=== LVM 状态 ==="
lvs
pvs

echo ""
echo "=== 磁盘空间 ==="
df -h /data
EOF
```

---

## 📊 两种方案对比

| 特性 | 方案 A（用户主目录） | 方案 B（修复 /data） |
|------|---------------------|---------------------|
| 是否需要 sudo | ❌ 不需要 | ✅ 需要 |
| 立即可用 | ✅ 是 | ⚠️ 需修复后 |
| 磁盘空间 | 主目录空间 | /data (7.3TB) |
| 数据安全性 | ✅ 高（独立） | ⚠️ 共享分区 |
| 长期维护 | ⚠️ 可能空间不足 | ✅ 更适合 |

---

## 🎯 推荐操作流程

### 今晚（立即执行）

```bash
# 1. 使用修复后的脚本立即开始评估
cd ~/AD_predict/AD_predict
./run_eval_fixed.sh

# 2. 退出（评估会在后台继续）
# 按 Ctrl+B，然后按 D

# 3. 休息去吧！明早查看结果
```

### 明天（查看结果）

```bash
# 1. 检查评估是否完成
tmux attach -t seniortalk_eval_fixed

# 2. 查看结果
cat ~/AD_predict_results/seniortalk_evaluation/comparison_table.md

# 3. 如果需要，修复 /data 分区以便后续使用
sudo ~/AD_predict/AD_predict/fix_data_partition.sh
```

---

## 📝 文件清单

### 新创建的文件

1. **评估脚本（修复版）**
   - `scripts/eval_seniortalk_available_models_fixed.py`
   - 使用用户主目录存储所有输出

2. **启动脚本（修复版）**
   - `run_eval_fixed.sh`
   - 在 tmux 中运行，使用 graph 虚拟环境

3. **修复脚本**
   - `fix_data_partition.sh`
   - 自动诊断和修复 /data 只读问题

4. **文档**
   - `FIX_DATA_READONLY.md` - 详细的修复指南
   - `COMPLETE_SOLUTION_GUIDE.md` - 本文件

### 原有文件（保持不变）

- `scripts/eval_seniortalk_available_models.py` - 原脚本
- `run_eval_now.sh` - 原启动脚本

---

## 🆘 常见问题

### Q1: 评估运行多久？

**A**: 预计 1-2 小时（取决于模型数量和样本数）

### Q2: 如何查看进度？

**A**: 
```bash
tail -f ~/AD_predict_logs/eval_seniortalk_*.log
```

### Q3: 磁盘空间不够怎么办？

**A**: 检查空间：
```bash
df -h ~
df -h /data
```

如果主目录空间不足，必须先修复 /data 分区

### Q4: 评估中断了怎么办？

**A**: 
```bash
# 重新运行
./run_eval_fixed.sh

# 或手动运行
conda activate graph
python scripts/eval_seniortalk_available_models_fixed.py
```

### Q5: 修复 /data 分区安全吗？

**A**: 
- `mount -o remount` 非常安全
- `e2fsck` 会先检查，然后才修复
- 建议先备份重要数据（如果有的话）

### Q6: 为什么会变成只读？

**A**: 常见原因：
- 文件系统错误自动触发只读保护
- 磁盘 I/O 错误
- 意外断电或异常关机

---

## 📞 需要帮助？

如果遇到其他问题，收集以下信息：

```bash
# 运行诊断
bash << 'EOF'
echo "=== 磁盘空间 ==="
df -h

echo ""
echo "=== /data 状态 ==="
mount | grep /data
stat /data

echo ""
echo "=== 评估脚本状态 ==="
tmux ls 2>/dev/null || echo "没有运行中的 tmux 会话"

echo ""
echo "=== 最近的日志 ==="
ls -lth ~/AD_predict_logs/ | head -n 5

echo ""
echo "=== Python 环境 ==="
which python
python --version
EOF
```

将输出保存并发给我！

---

## ✅ 总结

你现在有三个选择：

1. **立即开始评估**（推荐）
   ```bash
   cd ~/AD_predict/AD_predict && ./run_eval_fixed.sh
   ```

2. **先修复 /data，再使用原脚本**
   ```bash
   sudo ~/AD_predict/AD_predict/fix_data_partition.sh
   ```

3. **先诊断，再决定**
   ```bash
   cat ~/AD_predict/AD_predict/FIX_DATA_READONLY.md
   ```

**我的建议**：先运行方案 1，让评估跑起来，明天再处理 /data 的问题！🚀

