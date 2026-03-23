#!/bin/bash
#
# 在 tmux 中运行 SeniorTalk 评估
# 使用方法: bash scripts/run_seniortalk_eval_tmux.sh
#

SESSION_NAME="seniortalk_eval"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="/data/AD_predict/experiments/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/seniortalk_eval_${TIMESTAMP}.log"

echo "=================================="
echo "SeniorTalk Evaluation - Tmux Runner"
echo "=================================="
echo "Session name: $SESSION_NAME"
echo "Log file: $LOG_FILE"
echo "=================================="

# 检查 tmux 是否已安装
if ! command -v tmux &> /dev/null; then
    echo "错误: tmux 未安装"
    echo "请安装: sudo apt-get install tmux"
    exit 1
fi

# 检查会话是否已存在
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo ""
    echo "警告: tmux 会话 '$SESSION_NAME' 已存在"
    echo ""
    echo "选项:"
    echo "  1. 附加到现有会话: tmux attach -t $SESSION_NAME"
    echo "  2. 杀死现有会话: tmux kill-session -t $SESSION_NAME"
    echo ""
    read -p "是否杀死现有会话并创建新会话? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t $SESSION_NAME
        echo "已杀死现有会话"
    else
        echo "退出。请手动处理现有会话。"
        exit 0
    fi
fi

# 创建新的 tmux 会话并运行评估
echo ""
echo "创建 tmux 会话: $SESSION_NAME"
echo "启动评估脚本..."
echo ""

tmux new-session -d -s $SESSION_NAME

# 设置工作目录
tmux send-keys -t $SESSION_NAME "cd $PROJECT_ROOT" C-m

# 显示信息
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo 'SeniorTalk Model Evaluation'" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Started at: \$(date)'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Log file: $LOG_FILE'" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m

# 显示 GPU 信息
tmux send-keys -t $SESSION_NAME "nvidia-smi" C-m
tmux send-keys -t $SESSION_NAME "sleep 2" C-m

# 运行评估脚本
tmux send-keys -t $SESSION_NAME "python scripts/eval_seniortalk_available_models.py 2>&1 | tee $LOG_FILE" C-m

# 完成后显示消息
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Evaluation completed at: \$(date)'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Log saved to: $LOG_FILE'" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo '按 Ctrl+B 然后按 D 来分离会话'" C-m
tmux send-keys -t $SESSION_NAME "echo '使用 tmux attach -t $SESSION_NAME 重新附加'" C-m

echo "✓ Tmux 会话已创建并启动"
echo ""
echo "查看评估进度的方法:"
echo "  1. 附加到 tmux 会话:"
echo "     tmux attach -t $SESSION_NAME"
echo ""
echo "  2. 实时查看日志:"
echo "     tail -f $LOG_FILE"
echo ""
echo "  3. 列出所有 tmux 会话:"
echo "     tmux ls"
echo ""
echo "  4. 分离会话: Ctrl+B 然后按 D"
echo "  5. 杀死会话: tmux kill-session -t $SESSION_NAME"
echo ""
echo "=================================="
echo "评估已在后台启动!"
echo "=================================="
