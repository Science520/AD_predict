#!/bin/bash
#
# SeniorTalk 评估脚本启动器 - 镜像站版本
# 解决网络问题：使用 HF 镜像站
#

SESSION_NAME="seniortalk_eval_mirror"
SCRIPT_PATH="/home/saisai/AD_predict/AD_predict/scripts/eval_seniortalk_with_mirror.py"
LOG_DIR="$HOME/AD_predict_logs"
LOG_FILE="$LOG_DIR/eval_mirror_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 检查是否已存在会话
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠️  Session '$SESSION_NAME' already exists!"
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $SESSION_NAME"
    echo "  2. Kill existing session: tmux kill-session -t $SESSION_NAME"
    echo ""
    exit 1
fi

echo "================================================"
echo "Starting SeniorTalk Evaluation (Mirror Version)"
echo "================================================"
echo ""
echo "🌐 Network Configuration:"
echo "  - Using HF Mirror: hf-mirror.com"
echo "  - No VPN/Proxy required"
echo ""
echo "Session: $SESSION_NAME"
echo "Script:  $SCRIPT_PATH"
echo "Log:     $LOG_FILE"
echo ""
echo "All outputs will be saved to:"
echo "  - Results: ~/AD_predict_results/seniortalk_evaluation/"
echo "  - Cache:   ~/.cache/huggingface/"
echo "  - Logs:    $LOG_DIR/"
echo ""

# 创建新的 tmux 会话
tmux new-session -d -s $SESSION_NAME

# 发送命令
tmux send-keys -t $SESSION_NAME "cd /home/saisai/AD_predict/AD_predict" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Activating conda environment: graph'" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "source ~/miniconda3/etc/profile.d/conda.sh" C-m
tmux send-keys -t $SESSION_NAME "conda activate graph" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo 'Environment activated!'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Python:' \$(which python)" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Starting evaluation with HF Mirror...'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Mirror: hf-mirror.com'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Output: ~/AD_predict_results/'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Log: $LOG_FILE'" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "python $SCRIPT_PATH 2>&1 | tee $LOG_FILE" C-m

echo "✓ Tmux session started!"
echo ""
echo "Commands:"
echo "  - Attach:     tmux attach -t $SESSION_NAME"
echo "  - Detach:     Press Ctrl+B, then D"
echo "  - Kill:       tmux kill-session -t $SESSION_NAME"
echo "  - View log:   tail -f $LOG_FILE"
echo ""
echo "================================================"





