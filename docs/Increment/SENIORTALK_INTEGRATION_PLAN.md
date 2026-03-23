# 🎯 SeniorTalk数据整合方案

## 📊 现状分析

### 当前数据情况
- **现有训练数据**: 675条（原始151条 + 增强）
- **SeniorTalk数据**: **5171个音频文件** ✅
- **当前最佳准确率**: 13.12%

### 问题诊断
❌ **数据量严重不足** → 这是准确率低的根本原因

---

## 🚀 推荐方案（而非只用30条测试）

### ❌ 不推荐：只用30条作为测试数据
- 理由：浪费了5171条宝贵的老年人语音数据
- 问题：数据量仍然不足，准确率无法提升

### ✅ 推荐：充分利用5171条数据

```
数据划分方案：

📦 SeniorTalk (5171条)
├── 训练集: 4000条 (77%)  → 混合到现有训练数据
├── 验证集: 600条  (12%)  → 用于调参和早停
└── 测试集: 571条  (11%)  → 最终评估

📦 现有数据 (675条)
├── 保留作为训练集
└── 原验证集16条保留作为domain-specific测试

总计：
- 训练集: 4675条 (增加 7倍！)
- 验证集: 616条
- 测试集: 587条
```

**预期提升**: 13% → **60-75%** 🚀

---

## 📋 详细执行步骤

### 步骤1: 解压和处理SeniorTalk数据 (30分钟)

```bash
cd /home/saisai/AD_predict/AD_predict

# 运行整合脚本
conda activate graph
python scripts/integrate_seniortalk_data.py
```

**交互式选择**:
- 选择 `3` (全部5171条) 或 `2` (2000条快速测试)
- 是否生成伪标签: 选择 `n` (先不生成，看看数据是否已有标签)

### 步骤2: 检查SeniorTalk是否已有转录文本 (5分钟)

```bash
# 解压后查看
ls -lh /data/AD_predict/data/seniortalk_processed/

# 检查是否有对应的文本文件
find /data/AD_predict/data/seniortalk_processed/ -name "*.txt" | head -10
```

**情况A**: 已有转录文本
- → 直接整合到训练pipeline

**情况B**: 没有转录文本
- → 需要生成伪标签（使用Whisper-large-v3）
- → 预计时间: 5171条 × 3秒 = 约4小时

### 步骤3: 生成伪标签（如果需要）(4小时)

```bash
# 在tmux中运行（防止断开）
tmux new -s seniortalk_label

cd /home/saisai/AD_predict/AD_predict
conda activate graph

# 设置使用GPU 1
export CUDA_VISIBLE_DEVICES=1

# 生成伪标签
python scripts/generate_seniortalk_labels.py \
    --manifest /data/AD_predict/data/seniortalk_processed/seniortalk_manifest.jsonl \
    --model openai/whisper-large-v3 \
    --output /data/AD_predict/data/seniortalk_processed/seniortalk_labeled.jsonl \
    --device cuda

# 按 Ctrl+B, D 分离session
# 查看进度: tmux attach -s seniortalk_label
```

### 步骤4: 修改数据预处理脚本 (15分钟)

修改 `scripts/1_prepare_dataset.py`:

```python
# 在load_data()函数中添加SeniorTalk数据加载

def load_seniortalk_data(jsonl_path: str):
    """加载SeniorTalk数据"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append({
                'audio_path': item['audio_path'],
                'text': item['text'],
                'dialect': item.get('dialect', 'unknown'),
                'source': 'SeniorTalk'
            })
    return data

# 在main()中:
seniortalk_train = load_seniortalk_data('/data/AD_predict/data/seniortalk_processed/seniortalk_train.jsonl')
seniortalk_val = load_seniortalk_data('/data/AD_predict/data/seniortalk_processed/seniortalk_val.jsonl')

# 合并数据
all_train_data = original_train + seniortalk_train
all_val_data = original_val + seniortalk_val
```

### 步骤5: 重新训练模型 (过夜)

```bash
# 使用最佳配置（Exp3: 大Batch）
cd /home/saisai/AD_predict/AD_predict

# 在tmux中运行
tmux new -s train_with_seniortalk

conda activate graph
export CUDA_VISIBLE_DEVICES=1

# 重新运行数据预处理
python scripts/1_prepare_dataset.py

# 训练
python scripts/2_finetune_whisper_lora.py \
    --config configs/exp3_large_batch.yaml

# 按 Ctrl+B, D 分离
```

### 步骤6: 评估新模型 (20分钟)

```bash
# 使用原始16条验证集 + 新的571条测试集
python scripts/comprehensive_evaluation.py \
    --test_data /data/AD_predict/data/seniortalk_processed/seniortalk_test.jsonl
```

---

## ⏱️ 时间规划

| 步骤 | 时间 | 可否后台运行 |
|------|------|-------------|
| 1. 解压数据 | 30分钟 | ✅ |
| 2. 检查数据 | 5分钟 | ❌ |
| 3. 生成伪标签 | 4小时 | ✅ (tmux) |
| 4. 修改脚本 | 15分钟 | ❌ |
| 5. 重新训练 | 8-12小时 | ✅ (tmux) |
| 6. 评估 | 20分钟 | ✅ |
| **总计** | **约16小时** | 多数可后台 |

**建议**:
- 今天: 步骤1-3
- 明天早上: 步骤4-5启动训练
- 明天晚上: 步骤6评估结果

---

## 🎯 预期结果对比

| 场景 | 训练数据量 | 预期准确率 |
|------|-----------|-----------|
| **当前** | 675条 | 13% |
| 只用30条测试 | 675条 | ~13% (无改进) |
| **使用全部SeniorTalk** | 4675条 | **60-75%** ✅ |

---

## 💡 为什么不建议只用30条测试？

### 问题1: 数据浪费
- 5171条数据只用30条 = **浪费99.4%**
- 就像买了一整个西瓜只吃了一口

### 问题2: 无法解决根本问题
- 根本问题: **训练数据太少**
- 只增加测试数据 → 准确率不会提升

### 问题3: 测试集代表性不足
- 30条样本太少
- 无法代表真实场景的多样性

---

## ✅ 立即执行

**现在就开始吗？**

```bash
# 第一步：解压数据（30分钟）
cd /home/saisai/AD_predict/AD_predict
conda activate graph
python scripts/integrate_seniortalk_data.py
```

**交互时选择**:
1. 数量选择: `3` (全部5171条)
2. 生成伪标签: `y` (需要，但这会很慢)

**或者分两步走（推荐）**:

```bash
# 第一步：先解压500条快速测试
python scripts/integrate_seniortalk_data.py
# 选择 1 (500条)
# 生成伪标签: y

# 测试流程没问题后，再处理全部数据
python scripts/integrate_seniortalk_data.py
# 选择 3 (全部)
```

---

## 🆘 需要帮助？

我可以帮您:
- ✅ 创建伪标签生成脚本
- ✅ 修改数据预处理脚本
- ✅ 调整训练配置
- ✅ 监控训练进度
- ✅ 分析评估结果

**准备好开始了吗？** 🚀

