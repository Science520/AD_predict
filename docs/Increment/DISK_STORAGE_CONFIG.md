# 磁盘存储配置说明

## 硬盘分区情况

### 1. 主系统盘 (nvme1n1p2)
- **容量**: 915G
- **已用**: 644G (75%)
- **剩余**: 225G
- **挂载点**: `/`
- **用途**: 系统文件、代码、小文件

### 2. 大容量数据盘 (nvme0n1p2)
- **容量**: 1.8T
- **已用**: 901G (52%)
- **剩余**: 839G
- **挂载点**: `/mnt/nvme0n1p2`
- **用途**: 备用大文件存储

### 3. 超大容量存储 (vg_data-lv_data) ⭐ **推荐**
- **容量**: 7.3T
- **已用**: 221G (4%)
- **剩余**: 6.7T
- **挂载点**: `/data`
- **用途**: 主要数据存储

## 文件迁移情况

### 已迁移的大文件目录
以下目录已从项目根目录移动到 `/data/AD_predict/`：

1. **data/** (9.9G) → `/data/AD_predict/data/`
   - 原始音频数据
   - 处理后的数据
   - 标注文件

2. **processed_data/** (1.9G) → `/data/AD_predict/processed_data/`
   - 预处理后的数据集
   - 训练/验证/测试分割

3. **experiments/** (400K) → `/data/AD_predict/experiments/`
   - 实验配置
   - 评估结果
   - 日志文件

4. **logs/** (100K) → `/data/AD_predict/logs/`
   - 训练日志
   - TensorBoard事件文件

### 符号链接
为了保持项目结构不变，在项目根目录创建了符号链接：
```bash
data -> /data/AD_predict/data
processed_data -> /data/AD_predict/processed_data
experiments -> /data/AD_predict/experiments
logs -> /data/AD_predict/logs
```

## 配置文件更新

### 已更新的配置文件
1. `configs/training_args.yaml`
   - Excel文件路径: `/data/AD_predict/data/raw/audio/...`
   - 音频目录路径: `/data/AD_predict/data/raw/audio/elderly_audios`

2. `scripts/0_validate_data.py`
   - 默认路径更新为 `/data/AD_predict/data/...`

3. `data_utils/audio_augment.py`
   - 测试音频路径更新

4. `scripts/whisper_data_collection/1_analyze_and_sample.py`
   - Excel和音频目录路径更新

5. `scripts/whisper_data_collection/3_scrape_subtitles.py`
   - Excel文件路径更新

## 使用说明

### 1. 正常运行项目
由于使用了符号链接，项目的所有脚本和代码都可以正常运行，无需修改。

### 2. 访问数据文件
```bash
# 通过符号链接访问（推荐）
ls /home/saisai/AD_predict/AD_predict/data/

# 直接访问数据盘
ls /data/AD_predict/data/
```

### 3. 添加新的数据文件
```bash
# 将新的大文件直接放到数据盘
cp new_large_file.dat /data/AD_predict/data/

# 或在项目目录中创建符号链接
ln -s /data/AD_predict/data/new_large_file.dat /home/saisai/AD_predict/AD_predict/
```

### 4. 模型文件存储
模型文件建议存储在：
```bash
/data/AD_predict/models/
```

## 空间节省效果

- **主系统盘释放空间**: 约12G
- **主系统盘使用率**: 从76%降至75%
- **数据盘使用率**: 仅4%，有充足空间用于未来扩展

## 注意事项

1. **备份**: 重要数据已迁移到独立的数据盘，建议定期备份
2. **权限**: 所有文件权限已正确设置，用户 `saisai` 拥有完全访问权限
3. **路径**: 所有硬编码路径已更新，但建议使用相对路径或配置文件
4. **监控**: 建议定期检查磁盘空间使用情况

## 未来扩展

当需要更多存储空间时，可以考虑：
1. 使用 `/mnt/nvme0n1p2` (839G可用)
2. 扩展 `/data` 分区 (6.7T可用)
3. 添加新的硬盘分区
