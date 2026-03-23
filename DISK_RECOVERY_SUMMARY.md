# 硬盘修复与数据恢复总结

## 硬件诊断结果

### 损坏的存储设备
1. **`/dev/sda` (3.6T)** - ❌ 硬件故障
   - 错误：`DID_BAD_TARGET`, I/O error
   - LVM 卷组 `vg_data` 的一半，无法恢复
   - 状态：物理损坏或断连

2. **`/dev/nvme0n1p2` (1.8T)** - ⚠️ 文件系统损坏后已修复
   - 错误：EXT4 inode #2 (根目录) 损坏
   - 修复后状态：✅ 可读写，839G 可用
   - 挂载点：`/mnt/nvme0n1p2`

### 可用的存储设备
1. **`/dev/sdc1` (2.7T)** - ✅ 健康
   - 可用空间：2.2T
   - 挂载点：`/mnt/backup`
   - **包含完整备份数据！**

2. **`/dev/nvme1n1p2` (915G)** - ✅ 系统盘
   - 可用空间：39G (96% 满)
   - 挂载点：`/` (根目录)

3. **`/tmp`** - ✅ 临时文件系统
   - 用于 Hugging Face 缓存

---

## 数据恢复情况

### SeniorTalk 数据集 ✅ 完整恢复
**位置**: `/mnt/backup/data_backup/AD_predict/data/raw/seniortalk_full/`

- **sentence_data**: 8.0GB
  - ✅ test-00000-of-00003.parquet (284M)
  - ✅ test-00001-of-00003.parquet (270M)
  - ✅ test-00002-of-00003.parquet (271M)
  - ✅ train-*.parquet (16 files, ~6.5GB)
  - ✅ dev-*.parquet (2 files, ~900MB)

- **dialogue_data**: 12GB

### 训练模型 ✅ 完整恢复
**位置**: `/mnt/backup/data_backup/AD_predict/`

| 模型 | Checkpoint | 状态 |
|------|-----------|------|
| exp1_high_rank | checkpoint-100 | ✅ |
| exp2_low_lr | checkpoint-1100 | ✅ |
| exp3_large_batch | checkpoint-750 | ✅ |
| exp4_aggressive | checkpoint-500 | ✅ |
| whisper_lora_dialect | final_adapter | ✅ |

---

## 已完成的修复操作

### 1. 软链接重建
```bash
cd ~/AD_predict/AD_predict
ln -sf /mnt/backup/data_backup/AD_predict/data data
```
- 旧链接（指向损坏的 `/data`）已备份为 `data.old.broken`
- 新链接指向备份盘

### 2. 脚本路径更新
- ✅ `scripts/eval_seniortalk_available_models.py`
  - 数据路径：通过软链接 `data/` 访问
  - 模型路径：指向 `/mnt/backup/data_backup/AD_predict/exp*`
  - HF 缓存：`/tmp/saisai_hf_cache`
  - 输出目录：`~/AD_predict_results/seniortalk_evaluation`

- ✅ `scripts/eval_seniortalk_comprehensive.py`
  - 支持在线/本地两种模式
  - HF 缓存：`~/.cache/huggingface`（已修复损坏的软链接）

### 3. 启动脚本
- ✅ `run_eval_local.sh` - 使用本地备份数据
- ✅ `run_eval_online.sh` - 使用 HF 在线数据（备用）

---

## 当前存储策略

| 数据类型 | 位置 | 容量 | 说明 |
|---------|------|------|------|
| 原始数据集 | `/mnt/backup` (sdc1) | 20GB | 只读访问 |
| 训练模型 | `/mnt/backup` (sdc1) | ~数GB | 只读访问 |
| HF 缓存 | `/tmp` | 动态 | 临时，重启后清空 |
| 评估输出 | `~/AD_predict_results` | <100MB | 小文件 |

---

## 运行评估

### 快速启动
```bash
cd ~/AD_predict/AD_predict
./run_eval_local.sh
```

### 手动启动
```bash
conda activate graph
export HF_HOME=/tmp/saisai_hf_cache
export TRANSFORMERS_CACHE=/tmp/saisai_hf_cache/transformers
export HF_DATASETS_CACHE=/tmp/saisai_hf_cache/datasets
python scripts/eval_seniortalk_available_models.py
```

### 查看结果
```bash
cat ~/AD_predict_results/seniortalk_evaluation/comparison_table.md
ls -lh ~/AD_predict_results/seniortalk_evaluation/
```

---

## 注意事项

1. **HF 缓存在 /tmp**
   - 重启后会清空
   - 首次运行需重新下载 Whisper 模型（~1.5GB）
   - 如需持久化，可改为 `/mnt/nvme0n1p2/saisai_cache`（需 sudo 创建）

2. **备份盘只读**
   - `/mnt/backup` 根目录无写权限
   - 所有输出写入 `~/AD_predict_results`

3. **系统盘空间紧张**
   - `/` 只剩 39G
   - 定期清理 conda 缓存和日志

---

## 硬件建议

1. **立即**：更换或移除 `/dev/sda`（硬件故障）
2. **短期**：监控 `/dev/nvme0n1p2` 健康状态（曾损坏）
3. **长期**：扩容系统盘或迁移数据到 `/mnt/nvme0n1p2`

---

生成时间：2025-11-01

