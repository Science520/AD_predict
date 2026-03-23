#!/bin/bash
# 在tmux中运行评估脚本，确保SSH断开后继续运行

echo "================================================"
echo "🖥️  在tmux中启动CPU评估"
echo "================================================"

# 检查tmux是否安装
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux未安装！请先安装: sudo apt-get install tmux"
    exit 1
fi

# 定义会话名称
SESSION_NAME="whisper_eval"

# 检查会话是否已存在
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠️  会话 '$SESSION_NAME' 已存在"
    echo ""
    echo "选项："
    echo "  1. 连接到现有会话: tmux attach -t $SESSION_NAME"
    echo "  2. 杀掉旧会话重新开始: tmux kill-session -t $SESSION_NAME && bash $0"
    echo ""
    read -p "是否杀掉旧会话重新开始？(y/n): " answer
    if [ "$answer" = "y" ]; then
        echo "正在停止旧会话..."
        tmux kill-session -t $SESSION_NAME
        sleep 2
    else
        echo "退出。使用 'tmux attach -t $SESSION_NAME' 连接现有会话"
        exit 0
    fi
fi

# 停止可能正在运行的eval_cpu进程
echo "检查并停止旧的评估进程..."
pkill -f "python.*eval_cpu.py" 2>/dev/null
sleep 2

# 创建新的tmux会话并在其中运行评估
echo "创建tmux会话: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME

# 在tmux会话中设置环境和运行脚本
tmux send-keys -t $SESSION_NAME "cd /home/saisai/AD_predict/AD_predict" C-m
tmux send-keys -t $SESSION_NAME "conda activate graph" C-m
tmux send-keys -t $SESSION_NAME "export HF_HOME=~/.cache/huggingface" C-m
tmux send-keys -t $SESSION_NAME "export HF_ENDPOINT='https://hf-mirror.com'" C-m
tmux send-keys -t $SESSION_NAME "export HF_HUB_OFFLINE=1" C-m
tmux send-keys -t $SESSION_NAME "echo '================================================'" C-m
tmux send-keys -t $SESSION_NAME "echo '🚀 开始CPU评估'" C-m
tmux send-keys -t $SESSION_NAME "echo '================================================'" C-m
tmux send-keys -t $SESSION_NAME "python scripts/eval_cpu.py 2>&1 | tee /tmp/cpu_eval_tmux.log" C-m

echo ""
echo "✅ tmux会话已启动！"
echo ""
echo "================================================"
echo "📋 使用说明"
echo "================================================"
echo ""
echo "1️⃣  连接到会话查看进度："
echo "   tmux attach -t $SESSION_NAME"
echo ""
echo "2️⃣  从会话中分离（不停止程序）："
echo "   按键: Ctrl+B 然后按 D"
echo ""
echo "3️⃣  查看日志："
echo "   tail -f /tmp/cpu_eval_tmux.log"
echo ""
echo "4️⃣  列出所有tmux会话："
echo "   tmux list-sessions"
echo ""
echo "5️⃣  停止会话："
echo "   tmux kill-session -t $SESSION_NAME"
echo ""
echo "================================================"
echo "✨ 现在可以安全关闭电脑，任务会继续运行！"
echo "================================================"
echo ""
echo "提示: 5秒后自动连接到会话..."
sleep 5
tmux attach -t $SESSION_NAME

