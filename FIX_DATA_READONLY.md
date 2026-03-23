# 解决 /data 分区只读问题

## 🔍 问题诊断

从前面的检查结果来看：
- `/data` 显示为 `rw`（读写）挂载
- 但实际写入时报错 "Read-only file system"
- 文件系统可能因异常进入了只读保护模式

## 📋 诊断步骤

### 1. 检查系统日志（需要 sudo）

```bash
sudo dmesg | grep -iE "error|fail|readonly" | tail -n 50
```

### 2. 检查文件系统状态

```bash
sudo tune2fs -l /dev/mapper/vg_data-lv_data | grep -i "state"
```

### 3. 检查挂载选项

```bash
cat /proc/mounts | grep /data
```

## 🔧 解决方案

### 方案 1：重新挂载为读写（推荐优先尝试）

```bash
# 重新挂载为读写模式
sudo mount -o remount,rw /data

# 验证
touch /data/test_write_$(date +%s).txt && echo "✓ Write test passed!" || echo "✗ Still read-only"
```

### 方案 2：卸载后重新挂载

⚠️ **警告：此操作会断开所有对 /data 的访问，请确保没有程序正在使用 /data**

```bash
# 1. 检查谁在使用 /data
sudo lsof +D /data | head -n 20

# 2. 如果有 Docker 容器，先停止
sudo docker ps | grep -q . && sudo docker stop $(sudo docker ps -q)

# 3. 卸载
sudo umount /data

# 4. 运行文件系统检查（可选但推荐）
sudo e2fsck -f -y /dev/mapper/vg_data-lv_data

# 5. 重新挂载
sudo mount /data

# 6. 验证
df -h /data
touch /data/test_write_$(date +%s).txt && echo "✓ Write test passed!" || echo "✗ Still read-only"
```

### 方案 3：强制文件系统检查（如果方案 1 和 2 都失败）

⚠️ **警告：此操作需要卸载文件系统**

```bash
# 1. 停止所有使用 /data 的进程
sudo fuser -km /data

# 2. 卸载
sudo umount -f /data

# 3. 强制检查并修复
sudo e2fsck -f -y -v /dev/mapper/vg_data-lv_data

# 4. 重新挂载
sudo mount /data

# 5. 验证
df -h /data
ls -la /data/AD_predict/
```

## 🚀 快速修复脚本

我已为你创建了一个快速修复脚本：

```bash
sudo bash /home/saisai/AD_predict/AD_predict/fix_data_partition.sh
```

## 📌 临时解决方案（已完成）

在修复 `/data` 之前，你可以使用修改后的脚本，它将所有输出保存到用户主目录：

```bash
# 启动评估（使用用户主目录）
./run_eval_fixed.sh

# 查看日志
tail -f ~/AD_predict_logs/eval_seniortalk_*.log

# 查看结果
ls -lh ~/AD_predict_results/seniortalk_evaluation/
```

## 🔍 进一步诊断

如果上述方案都无法解决，运行：

```bash
# 完整诊断报告
sudo bash << 'EOF'
echo "=== Filesystem State ==="
tune2fs -l /dev/mapper/vg_data-lv_data | grep -E "state|error|mount"

echo ""
echo "=== Mount Status ==="
mount | grep /data

echo ""
echo "=== Disk Errors ==="
dmesg | grep -iE "/data|vg_data|error" | tail -n 30

echo ""
echo "=== LVM Status ==="
lvs
pvs

echo ""
echo "=== File System Check (read-only) ==="
e2fsck -n /dev/mapper/vg_data-lv_data | head -n 20
EOF
```

## 📝 注意事项

1. **备份重要数据**：在进行任何修复操作前，确认重要数据已备份
2. **停止服务**：卸载前确保没有程序正在使用 /data
3. **Docker 影响**：如果 Docker 容器正在运行，可能需要先停止
4. **I/O 错误**：如果看到硬件 I/O 错误，可能是磁盘物理问题，需要联系管理员

## ✅ 验证修复

修复后运行这些命令验证：

```bash
# 1. 创建测试文件
echo "test" | sudo tee /data/test_write_$(date +%s).txt

# 2. 创建测试目录
sudo mkdir -p /data/test_dir_$(date +%s)

# 3. 检查模型文件
ls -lh /data/AD_predict/exp1_high_rank/checkpoint-100/

# 4. 检查磁盘状态
df -h /data
sudo tune2fs -l /dev/mapper/vg_data-lv_data | grep -i state
```

## 🎯 推荐执行顺序

1. **先尝试方案 1**（最简单，无风险）
2. 如果失败，查看完整诊断报告
3. 根据诊断结果决定是否使用方案 2 或 3
4. 如果都失败，保存诊断报告并联系系统管理员

