#!/bin/bash
# Whisper微调训练启动脚本（GPU 1优化版）
# 功能：清理GPU进程，设置环境变量，启动训练

set -e  # 遇到错误立即退出

echo "========================================"
echo "Whisper微调训练启动脚本"
echo "========================================"

# 1. 清理旧的训练进程
echo "步骤 1: 清理旧的训练进程..."
pkill -f "python scripts/2_finetune_whisper_lora.py" || true
sleep 2
echo "✓ 旧进程已清理"

# 2. 清理GPU 1缓存
echo "步骤 2: 清理GPU缓存..."
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
echo "✓ GPU缓存已清理"

# 3. 检查GPU状态
echo "步骤 3: 检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo ""

# 4. 设置环境变量
echo "步骤 4: 设置环境变量..."
export CUDA_VISIBLE_DEVICES=1  # 只使用GPU 1
export HF_HOME=~/.cache/huggingface
export HF_ENDPOINT='https://hf-mirror.com'
export HF_HUB_OFFLINE=1  # 离线模式

# PyTorch内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 减少内存碎片

echo "✓ 环境变量已设置:"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

# 5. 选择配置文件（默认使用优化的large配置）
CONFIG_FILE="${1:-configs/training_args_large_optimized.yaml}"
echo ""
echo "步骤 5: 加载配置..."
echo "  配置文件: $CONFIG_FILE"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 6. 启动训练
echo ""
echo "========================================"
echo "开始训练..."
echo "========================================"
echo "日志文件: /tmp/whisper_training_$(date +%Y%m%d_%H%M%S).log"
echo ""

cd /home/saisai/AD_predict/AD_predict
conda activate graph

LOG_FILE="/tmp/whisper_training_$(date +%Y%m%d_%H%M%S).log"

# 启动训练并记录日志
python scripts/2_finetune_whisper_lora.py \
    --config "$CONFIG_FILE" \
    2>&1 | tee "$LOG_FILE"

# 训练完成
echo ""
echo "========================================"
echo "训练完成！"
echo "========================================"
echo "日志文件: $LOG_FILE"
echo ""
echo "查看TensorBoard:"
echo "  tensorboard --logdir=/data/AD_predict/logs_large_optimized"

