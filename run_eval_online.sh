#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "SeniorTalk 在线评估脚本"
echo "=========================================="
echo

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graph

# 设置环境变量
export USE_ONLINE_DATASET=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface
export HF_DATASETS_CACHE=~/.cache/huggingface

echo "环境配置:"
echo "  - Conda env: graph"
echo "  - 数据源: Hugging Face 在线 (镜像站)"
echo "  - 输出目录: ~/AD_predict_results/seniortalk_evaluation"
echo "  - 缓存目录: ~/.cache/huggingface"
echo

# 运行评估
echo "开始评估..."
python scripts/eval_seniortalk_available_models.py

echo
echo "=========================================="
echo "✓ 评估完成"
echo "=========================================="
echo "结果位置: ~/AD_predict_results/seniortalk_evaluation/"

