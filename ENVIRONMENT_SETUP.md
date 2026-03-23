# SeniorTalk 评估环境配置指南

## 问题总结

在运行 `eval_seniortalk_available_models.py` 时遇到的依赖问题：

1. ✅ **torchcodec 损坏** - 已卸载，使用 soundfile/librosa 替代
2. ✅ **网络连接失败** - 已配置 HF 镜像站
3. ✅ **tabulate 缺失** - 已安装

## 快速修复

### 方法 1：自动设置（推荐）

```bash
cd ~/AD_predict/AD_predict
./setup_eval_env.sh
```

### 方法 2：手动安装

```bash
# 激活环境
conda activate graph

# 卸载问题包
pip uninstall -y torchcodec

# 安装所有依赖
pip install -r requirements_eval.txt

# 验证安装
python -c "import soundfile, librosa, tabulate, jiwer; print('✓ All dependencies installed')"
```

## 核心依赖列表

### 必需包（已安装）
- ✅ torch, torchaudio
- ✅ transformers, peft, accelerate
- ✅ datasets, pandas, numpy
- ✅ soundfile, librosa (音频解码)
- ✅ jiwer, pypinyin (评估指标)
- ✅ tqdm, pyarrow, tabulate (工具)

### 已移除包
- ❌ torchcodec (有依赖冲突，已卸载)

## 运行评估

```bash
cd ~/AD_predict/AD_predict

# 使用便捷脚本
./run_eval_local.sh

# 或手动运行
export HF_HOME=/tmp/saisai_hf_cache
export HF_ENDPOINT=https://hf-mirror.com
python scripts/eval_seniortalk_available_models.py
```

## 环境变量说明

```bash
# Hugging Face 缓存目录（避免 /data 权限问题）
export HF_HOME=/tmp/saisai_hf_cache
export TRANSFORMERS_CACHE=/tmp/saisai_hf_cache/transformers
export HF_DATASETS_CACHE=/tmp/saisai_hf_cache/datasets

# Hugging Face 镜像站（解决网络问题）
export HF_ENDPOINT=https://hf-mirror.com
```

## 故障排除

### 问题：`ImportError: To support decoding audio data, please install 'torchcodec'`

**原因：** `torchcodec` 包损坏或 FFmpeg 依赖缺失

**解决：**
```bash
pip uninstall -y torchcodec
pip install soundfile librosa
```

### 问题：`ImportError: Missing optional dependency 'tabulate'`

**解决：**
```bash
pip install tabulate
```

### 问题：网络连接 huggingface.co 失败

**解决：**
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题：模型维度不匹配 (size mismatch)

**原因：** 备份盘中的模型是用 `whisper-large` 训练的，但脚本配置为 `whisper-medium`

**解决：** 检查模型配置文件或跳过该模型

## 数据和模型位置

- **数据集：** `/mnt/backup/data_backup/AD_predict/data` (通过软链接 `data/`)
- **模型：** `/mnt/backup/data_backup/AD_predict/exp*/`
- **输出：** `~/AD_predict_results/seniortalk_evaluation/`
- **缓存：** `/tmp/saisai_hf_cache/`

## 完整依赖列表

详见 `requirements_eval.txt`

```bash
cat requirements_eval.txt
```

