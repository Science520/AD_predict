#!/bin/bash
#
# 立即启动 SeniorTalk 评估 - 在 tmux 后台运行
# 所有7个模型都可用！
#

SESSION_NAME="seniortalk_eval"
LOG_FILE="/home/saisai/AD_predict/AD_predict/logs/seniortalk_eval_$(date +%Y%m%d_%H%M%S).log"
mkdir -p /home/saisai/AD_predict/AD_predict/logs

echo "========================================"
echo "SeniorTalk 模型评估 - Tmux 启动"
echo "========================================"
echo "会话名: $SESSION_NAME"
echo "日志: $LOG_FILE"
echo ""
echo "评估模型数量: 7个"
echo "  1. Whisper-Medium 原始基线"
echo "  2. Exp1: High Rank"
echo "  3. Exp2: Low LR"
echo "  4. Exp3: Large Batch"
echo "  5. Exp4: Aggressive LR"
echo "  6. Best Model"
echo "  7. Dialect Final"
echo ""
echo "测试样本: 100个 (从SeniorTalk test set)"
echo "========================================"
echo ""

# 检查tmux
if ! command -v tmux &> /dev/null; then
    echo "错误: tmux 未安装"
    exit 1
fi

# 如果会话已存在，询问
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "警告: 会话 '$SESSION_NAME' 已存在"
    echo ""
    read -p "杀死现有会话并创建新会话? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t $SESSION_NAME
    else
        echo "退出。使用 'tmux attach -t $SESSION_NAME' 附加到现有会话"
        exit 0
    fi
fi

# 创建新会话
echo "创建 tmux 会话..."
tmux new-session -d -s $SESSION_NAME

# 激活 graph 虚拟环境并进入项目目录
tmux send-keys -t $SESSION_NAME "cd /home/saisai/AD_predict/AD_predict" C-m
tmux send-keys -t $SESSION_NAME "source ~/miniconda3/etc/profile.d/conda.sh" C-m
tmux send-keys -t $SESSION_NAME "conda activate graph" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo 'SeniorTalk 模型评估'" C-m
tmux send-keys -t $SESSION_NAME "echo '虚拟环境: graph'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Python: \$(which python)'" C-m
tmux send-keys -t $SESSION_NAME "echo '开始时间: \$(date)'" C-m
tmux send-keys -t $SESSION_NAME "echo '日志: $LOG_FILE'" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m

# 显示GPU
tmux send-keys -t $SESSION_NAME "nvidia-smi" C-m
tmux send-keys -t $SESSION_NAME "sleep 3" C-m

# 运行评估
tmux send-keys -t $SESSION_NAME "python scripts/eval_seniortalk_available_models.py 2>&1 | tee $LOG_FILE" C-m

# 完成提示
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo '评估完成: \$(date)'" C-m
tmux send-keys -t $SESSION_NAME "echo '日志: $LOG_FILE'" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m

echo "✓ Tmux 会话已创建并启动"
echo ""
echo "查看进度:"
echo "  1. 附加到会话: tmux attach -t $SESSION_NAME"
echo "  2. 查看日志: tail -f $LOG_FILE"
echo "  3. 列出会话: tmux ls"
echo ""
echo "分离会话: Ctrl+B 然后按 D"
echo "杀死会话: tmux kill-session -t $SESSION_NAME"
echo ""
echo "========================================"
echo "评估已在后台启动，可以休息了！"
echo "========================================"

