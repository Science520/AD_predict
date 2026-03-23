#!/bin/bash

# 内存优化的训练启动脚本
echo "🚀 启动内存优化的Whisper LoRA训练..."

# 设置环境变量
export HF_HOME=~/.cache/huggingface
export HF_ENDPOINT='https://hf-mirror.com'
export HF_HUB_OFFLINE=1

# 设置PyTorch内存管理
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 清理GPU内存
echo "🧹 清理GPU内存..."
python -c "import torch; torch.cuda.empty_cache(); print('GPU缓存已清理')"

# 检查GPU状态
echo "📊 当前GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graph

# 设置CUDA设备（使用GPU 1，避开Isaac Sim占用的GPU 0）
export CUDA_VISIBLE_DEVICES=1  # 使用GPU 1

# 启动训练
echo "🎯 开始训练..."
cd /home/saisai/AD_predict/AD_predict

python scripts/2_finetune_whisper_lora.py \
    --config configs/training_args.yaml \
    --resume_from_checkpoint "" \
    2>&1 | tee /data/AD_predict/training_optimized.log

echo "✅ 训练完成或中断"
