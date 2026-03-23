# 🚀 SeniorTalk评估快速开始

## 📋 评估说明

**目的**: 使用SeniorTalk老年人语音数据集评估您的4个微调模型和原始Whisper-medium模型

**策略**: 
1. 从SeniorTalk tar包中随机提取50个音频样本
2. 使用**Whisper-large-v3**生成高质量转录作为"参考标准"
3. 用5个模型分别转录这50个样本
4. 对比每个模型的预测结果与参考标准，计算准确率

**为什么这样做**:
- ✅ SeniorTalk数据没有现成的转录文本
- ✅ Whisper-large-v3比medium更强大，可以作为高质量参考
- ✅ 这样可以评估微调模型相比基线的提升
- ✅ 所有数据解压在`/data`目录，不占用系统盘

---

## 🎯 一键启动

### 方式1: 直接运行（前台）

```bash
cd /home/saisai/AD_predict/AD_predict
chmod +x scripts/run_seniortalk_eval.sh
bash scripts/run_seniortalk_eval.sh
```

**特点**: 
- 可以实时看到进度
- 需要保持SSH连接
- 预计30-60分钟

---

### 方式2: 后台运行（推荐）⭐

```bash
cd /home/saisai/AD_predict/AD_predict
chmod +x scripts/run_seniortalk_eval_tmux.sh
bash scripts/run_seniortalk_eval_tmux.sh
```

**特点**:
- ✅ 后台运行，可以关闭SSH
- ✅ 随时查看进度
- ✅ 意外断开也不受影响

**管理命令**:
```bash
# 查看实时进度
tmux attach -t seniortalk_eval

# 分离会话（按键）
Ctrl+B, 然后按 D

# 查看日志
tail -f /data/AD_predict/seniortalk_evaluation/eval.log

# 停止评估
tmux kill-session -t seniortalk_eval
```

---

## 📊 评估流程

评估会自动执行以下步骤：

```
步骤1: 从tar包提取50个音频样本 (1-2分钟)
   └─> /data/AD_predict/data/seniortalk_eval/

步骤2: 用Whisper-large-v3生成参考转录 (5-10分钟)
   └─> /data/AD_predict/data/seniortalk_eval/reference_transcripts.json

步骤3: 评估5个模型 (20-40分钟)
   ├─> 原始Whisper-Medium
   ├─> Exp1: 高Rank
   ├─> Exp2: 低学习率
   ├─> Exp3: 大Batch ⭐
   └─> Exp4: 激进学习率

步骤4: 生成评估报告 (1分钟)
   └─> /data/AD_predict/seniortalk_evaluation/
```

---

## 📁 输出文件

评估完成后，结果保存在 `/data/AD_predict/seniortalk_evaluation/`:

```
/data/AD_predict/seniortalk_evaluation/
├── SENIORTALK_EVALUATION_REPORT.md    # 📝 详细报告（推荐阅读）
├── summary.csv                         # 📊 结果汇总表格
├── detailed_results.json               # 🔍 完整评估数据
└── eval.log                            # 📋 运行日志
```

---

## 🔍 查看结果

### 快速查看汇总

```bash
# 查看CSV表格
cat /data/AD_predict/seniortalk_evaluation/summary.csv

# 或者用column美化显示
column -t -s, /data/AD_predict/seniortalk_evaluation/summary.csv
```

### 查看详细报告

```bash
# 查看Markdown报告
cat /data/AD_predict/seniortalk_evaluation/SENIORTALK_EVALUATION_REPORT.md

# 或者在IDE中打开
code /data/AD_predict/seniortalk_evaluation/SENIORTALK_EVALUATION_REPORT.md
```

### 查看转录对比示例

报告中会包含每个样本的转录对比：

```
样本1: xxx.wav
参考转录 (Whisper-large-v3): "今天天气真不错"

原始Whisper-Medium (准确率: 75%): "今天天气真的不错"
Exp3: 大Batch (准确率: 92%): "今天天气真不错"
...
```

---

## ⚙️ 自定义配置

如果想修改评估样本数量，编辑 `scripts/eval_on_seniortalk.py`:

```python
# 第24行
NUM_SAMPLES = 50  # 改为 100、200 等

# 样本数越多，评估越准确，但时间越长
# 50个样本: ~30-60分钟
# 100个样本: ~60-120分钟
```

---

## ⏱️ 时间估算

| 步骤 | 时间 | 说明 |
|------|------|------|
| 解压音频 | 1-2分钟 | 50个样本 |
| 生成参考转录 | 5-10分钟 | Whisper-large-v3 |
| 评估模型1 | 3-5分钟 | 基线 |
| 评估模型2-5 | 12-20分钟 | 4个微调模型 |
| 生成报告 | <1分钟 | |
| **总计** | **30-60分钟** | 取决于GPU速度 |

---

## 💡 常见问题

### Q1: 为什么用Whisper-large-v3作为参考？

**A**: SeniorTalk没有人工标注的转录文本。Whisper-large-v3是目前最强的开源ASR模型，可以生成高质量的"伪标签"作为参考标准。这样我们可以评估微调模型是否接近或超越了大模型的性能。

### Q2: 只评估50个样本够吗？

**A**: 
- ✅ 对于快速评估：50个样本足够发现模型间的差异
- ✅ 如果需要更精确的评估，可以增加到100-200个
- ⚠️ 样本数越多，评估时间越长

### Q3: 如果评估中断了怎么办？

**A**: 
- 参考转录会保存在`/data/AD_predict/data/seniortalk_eval/reference_transcripts.json`
- 重新运行时会自动使用已有的参考转录，不会重复生成
- 只会重新评估模型

### Q4: 磁盘空间占用多少？

**A**:
- 50个音频样本: ~50-100MB
- 参考转录JSON: <1MB
- 所有输出文件: <5MB
- **总计**: ~100MB（都在`/data`盘）

---

## 🎯 立即开始

**准备好了吗？运行这个命令**:

```bash
cd /home/saisai/AD_predict/AD_predict && bash scripts/run_seniortalk_eval_tmux.sh
```

然后关闭电脑去喝杯咖啡，回来查看结果！☕

---

## 📞 需要帮助？

如果遇到问题，检查：
1. GPU是否可用: `nvidia-smi`
2. 磁盘空间: `df -h /data`
3. 日志文件: `tail -f /data/AD_predict/seniortalk_evaluation/eval.log`
4. tmux会话: `tmux ls`

