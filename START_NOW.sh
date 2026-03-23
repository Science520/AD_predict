#!/bin/bash
# 🚀 一键启动今晚实验

echo "========================================"
echo "🌙 准备启动今晚实验"
echo "========================================"
echo ""

cd /home/saisai/AD_predict/AD_predict

echo "检查数据增强状态..."
AUG_COUNT=$(ls /data/AD_predict/data/raw/audio/elderly_audios_augmented/*.wav 2>/dev/null | wc -l)
echo "增强音频文件: $AUG_COUNT 个"

if [ "$AUG_COUNT" -lt 500 ]; then
    echo "⚠️  增强音频不足，建议重新运行数据增强"
    echo "   python scripts/1_prepare_dataset.py"
    exit 1
fi

echo "✓ 数据准备完成"
echo ""

echo "选择实验方案:"
echo "  1) Whisper实验 (4组, 2-3小时) [推荐]"
echo "  2) Whisper+FunASR (8组, 4-5小时)"
echo ""
read -p "请选择 (1/2): " choice

case $choice in
    1)
        echo ""
        echo "启动Whisper实验..."
        nohup bash scripts/run_experiments.sh > /tmp/experiments.log 2>&1 &
        LOG_FILE="/tmp/experiments.log"
        ;;
    2)
        echo ""
        echo "启动Whisper+FunASR实验..."
        nohup bash scripts/run_all_experiments.sh > /tmp/all_experiments.log 2>&1 &
        LOG_FILE="/tmp/all_experiments.log"
        ;;
    *)
        echo "无效选择，退出"
        exit 1
        ;;
esac

sleep 3

echo ""
echo "========================================"
echo "✅ 实验已启动！"
echo "========================================"
echo ""
echo "开始时间: $(date)"
echo "日志文件: $LOG_FILE"
echo ""
echo "监控命令:"
echo "  tail -f $LOG_FILE"
echo "  bash CHECK_STATUS.sh"
echo "  nvidia-smi"
echo ""
echo "🌙 祝好梦！明早见！"
