#!/bin/bash

# SeniorTalk评估脚本启动器
# 使用GPU 1，在tmux中运行

echo "========================================"
echo "🎯 SeniorTalk数据集评估"
echo "========================================"

cd /home/saisai/AD_predict/AD_predict

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graph

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=~/.cache/huggingface
export HF_ENDPOINT='https://hf-mirror.com'
export HF_HUB_OFFLINE=1

echo ""
echo "📊 配置信息:"
echo "   - 使用GPU: 1"
echo "   - 评估样本数: 50个"
echo "   - 解压目录: /data/AD_predict/data/seniortalk_eval"
echo "   - 输出目录: /data/AD_predict/seniortalk_evaluation"
echo ""

# 检查tar包是否存在
if [ ! -f "/data/AD_predict/data/raw/audio/seniortalk_asr_single/sentence_data/wav/train/train-0001.tar" ]; then
    echo "❌ 错误: 找不到SeniorTalk tar包"
    exit 1
fi

echo "✅ 准备就绪，开始评估..."
echo ""

# 运行评估
python scripts/eval_on_seniortalk.py 2>&1 | tee /data/AD_predict/seniortalk_evaluation/eval.log

echo ""
echo "✅ 评估完成！"
echo ""
echo "📊 查看结果:"
echo "   cat /data/AD_predict/seniortalk_evaluation/SENIORTALK_EVALUATION_REPORT.md"
echo "   cat /data/AD_predict/seniortalk_evaluation/summary.csv"

