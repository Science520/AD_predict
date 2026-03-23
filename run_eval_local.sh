#!/bin/bash
#
# SeniorTalk 评估脚本启动器 - 本地模型版本
# 使用本地已有的模型，不下载，不复制
# 重要：不要用 sudo 运行此脚本！
#

# 检查是否用 sudo 运行
if [ "$EUID" -eq 0 ]; then
    echo "❌ 错误：不要用 sudo 运行此脚本！"
    echo ""
    echo "sudo 会导致所有输出写到 /root/ 而不是你的用户目录"
    echo ""
    echo "正确运行方式:"
    echo "  ./run_eval_local.sh"
    echo ""
    echo "或者:"
    echo "  bash run_eval_local.sh"
    echo ""
    exit 1
fi

SESSION_NAME="seniortalk_eval_local"
SCRIPT_PATH="/home/saisai/AD_predict/AD_predict/scripts/eval_seniortalk_local_models.py"
LOG_DIR="$HOME/AD_predict_logs"
LOG_FILE="$LOG_DIR/eval_local_$(date +%Y%m%d_%H%M%S).log"

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
echo "Starting SeniorTalk Evaluation (Local Models)"
echo "================================================"
echo ""
echo "💾 Using Local Models:"
echo "  - Whisper: /data/saisai/cache/whisper"
echo "  - No download required"
echo "  - No network needed"
echo ""
echo "📁 Output Locations:"
echo "  - Results: $HOME/AD_predict_results/"
echo "  - Logs:    $LOG_DIR/"
echo "  - Cache:   $HOME/.cache/huggingface/"
echo ""
echo "⚠️  Important: Running as user $USER (not root)"
echo ""
echo "Session: $SESSION_NAME"
echo "Script:  $SCRIPT_PATH"
echo "Log:     $LOG_FILE"
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
tmux send-keys -t $SESSION_NAME "echo 'User: '\$(whoami)" C-m
tmux send-keys -t $SESSION_NAME "echo 'Python: '\$(which python)" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Starting evaluation with local models...'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Model: /data/saisai/cache/whisper'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Output: $HOME/AD_predict_results/'" C-m
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
echo ""
echo "✅ 已以普通用户 ($USER) 身份启动"
echo "✅ 所有输出将保存到你的用户目录"
echo ""
