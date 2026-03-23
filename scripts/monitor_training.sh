#!/bin/bash
# Whisper训练实时监控脚本

echo "========================================"
echo "Whisper训练实时监控"
echo "========================================"
echo ""

# 获取最新的日志文件
LOG_FILE=$(ls -t /tmp/whisper_logs/whisper_medium_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ 未找到训练日志文件"
    exit 1
fi

echo "日志文件: $LOG_FILE"
echo ""

# 显示GPU状态
echo "GPU状态:"
echo "----------------------------------------"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
  awk -F', ' '{printf "GPU %s: %s\n  内存: %s / %s MB (%.1f%%)\n  利用率: %s%%\n\n", $1, $2, $3, $4, ($3/$4)*100, $5}'
echo "----------------------------------------"
echo ""

# 显示训练进度
echo "训练进度:"
echo "----------------------------------------"
# 获取最近的进度信息
tail -100 "$LOG_FILE" | grep -E "it/s|%\|" | tail -5

# 获取最近的Loss信息（如果有）
echo ""
echo "最近的训练Loss:"
tail -200 "$LOG_FILE" | grep -i "loss" | tail -3

echo "----------------------------------------"
echo ""

# 显示评估结果（如果有）
EVAL_RESULTS=$(tail -500 "$LOG_FILE" | grep -i "eval.*wer\|eval.*loss" | tail -3)
if [ -n "$EVAL_RESULTS" ]; then
    echo "评估结果:"
    echo "----------------------------------------"
    echo "$EVAL_RESULTS"
    echo "----------------------------------------"
    echo ""
fi

# 检查是否有错误
ERRORS=$(tail -100 "$LOG_FILE" | grep -i "error\|exception\|out of memory" | tail -3)
if [ -n "$ERRORS" ]; then
    echo "⚠️  检测到错误:"
    echo "----------------------------------------"
    echo "$ERRORS"
    echo "----------------------------------------"
    echo ""
fi

# 显示实时跟踪命令
echo "实时跟踪命令:"
echo "  tail -f $LOG_FILE"
echo ""
echo "查看完整日志:"
echo "  less $LOG_FILE"
echo ""
echo "持续监控（每10秒刷新）:"
echo "  watch -n 10 bash $0"


