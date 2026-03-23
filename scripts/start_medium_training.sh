#!/bin/bash
# Whisper-Medium微调训练启动脚本（GPU 1，最小内存版）
# 功能：杀掉旧进程，清理GPU，设置环境变量，启动训练

set -e  # 遇到错误立即退出

echo "========================================"
echo "Whisper-Medium微调训练启动脚本"
echo "========================================"

# 1. 杀掉所有旧的训练进程
echo "步骤 1: 清理旧的训练进程..."
pkill -f "python.*2_finetune_whisper_lora.py" || true
sleep 3
echo "✓ 旧进程已清理"

# 2. 清理GPU缓存
echo ""
echo "步骤 2: 清理GPU缓存..."
python3 -c "import torch; torch.cuda.empty_cache(); print('GPU缓存已清理')" 2>/dev/null || echo "无法清理GPU缓存（可能是因为没有torch）"
sleep 2

# 3. 检查GPU状态
echo ""
echo "步骤 3: 检查GPU状态..."
echo "----------------------------------------"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
  awk -F', ' '{printf "GPU %s: %s\n  内存: %s MB / %s MB (%.1f%%)\n  利用率: %s%%\n\n", $1, $2, $3, $4, ($3/$4)*100, $5}'
echo "----------------------------------------"

# 检查Isaac Sim是否占用GPU
ISAAC_PID=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep isaac-sim | awk -F',' '{print $1}' | head -1)
if [ -n "$ISAAC_PID" ]; then
    echo "⚠️  警告: 检测到Isaac Sim进程 (PID: $ISAAC_PID) 正在占用GPU"
    echo "   这可能会导致内存不足，建议在训练前关闭Isaac Sim"
    echo ""
fi

# 4. 设置环境变量
echo "步骤 4: 设置环境变量..."
export CUDA_VISIBLE_DEVICES=1  # 🔥 只使用GPU 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # 按PCI总线顺序排列GPU
export HF_HOME=~/.cache/huggingface
export HF_ENDPOINT='https://hf-mirror.com'
export HF_HUB_OFFLINE=1  # 离线模式

# PyTorch内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128  # 减少内存碎片
export OMP_NUM_THREADS=4  # 限制OpenMP线程数
export MKL_NUM_THREADS=4  # 限制MKL线程数

# 禁用不必要的调试功能
export NCCL_DEBUG=WARN  # 降低NCCL日志级别
export TOKENIZERS_PARALLELISM=false  # 禁用tokenizer并行（避免fork问题）

echo "✓ 环境变量已设置:"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo ""

# 5. 选择配置文件
CONFIG_FILE="${1:-configs/training_args_medium_minimal.yaml}"
echo "步骤 5: 加载配置..."
echo "  配置文件: $CONFIG_FILE"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_FILE"
    echo ""
    echo "可用的配置文件:"
    ls -lh configs/training_args*.yaml 2>/dev/null || echo "  无配置文件"
    exit 1
fi

# 6. 确认训练目录
TRAIN_DIR="/home/saisai/AD_predict/AD_predict"
if [ ! -d "$TRAIN_DIR" ]; then
    echo "❌ 错误: 训练目录不存在: $TRAIN_DIR"
    exit 1
fi

cd "$TRAIN_DIR"
echo "  工作目录: $(pwd)"
echo ""

# 7. 启动训练
echo "========================================"
echo "开始训练..."
echo "========================================"

# 创建日志文件
LOG_DIR="/tmp/whisper_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/whisper_medium_$(date +%Y%m%d_%H%M%S).log"

echo "日志文件: $LOG_FILE"
echo ""

# 激活conda环境并启动训练
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graph

# 显示Python和PyTorch版本
echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# 启动训练并记录日志
python scripts/2_finetune_whisper_lora.py \
    --config "$CONFIG_FILE" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

# 8. 训练结束处理
echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 训练完成！"
else
    echo "❌ 训练失败 (退出码: $EXIT_CODE)"
fi
echo "========================================"
echo ""
echo "日志文件: $LOG_FILE"
echo ""
echo "查看训练指标:"
echo "  tail -100 $LOG_FILE"
echo ""
echo "查看TensorBoard:"
echo "  tensorboard --logdir=/data/AD_predict/logs_medium_minimal"
echo ""


