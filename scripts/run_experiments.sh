#!/bin/bash
# 自动化运行5组对比实验
# 用于参数调优和性能对比

set -e

PROJECT_ROOT="/home/saisai/AD_predict/AD_predict"
cd "$PROJECT_ROOT"

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graph

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=~/.cache/huggingface
export HF_ENDPOINT='https://hf-mirror.com'
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# 实验配置列表
EXPERIMENTS=(
    "exp1_high_rank:configs/exp1_high_rank.yaml:实验1-高LoRA rank"
    "exp2_low_lr:configs/exp2_low_lr.yaml:实验2-低学习率"
    "exp3_large_batch:configs/exp3_large_batch.yaml:实验3-大batch"
    "exp4_aggressive:configs/exp4_aggressive.yaml:实验4-激进训练"
)

# 创建结果目录
RESULTS_DIR="/data/AD_predict/experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "Whisper方言ASR参数调优实验"
echo "========================================"
echo "开始时间: $(date)"
echo "结果目录: $RESULTS_DIR"
echo "实验数量: ${#EXPERIMENTS[@]}"
echo ""

# 记录baseline结果
echo "📊 Baseline结果（已完成）:"
echo "  配置: configs/training_args_medium_minimal.yaml"
echo "  WER: 0.9984"
echo "  训练时间: 27分钟"
echo ""

# 运行每个实验
TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}
CURRENT=1

for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name config_file desc <<< "$exp"
    
    echo "========================================"
    echo "[$CURRENT/$TOTAL_EXPERIMENTS] $desc"
    echo "========================================"
    echo "配置文件: $config_file"
    echo "输出目录: /data/AD_predict/$exp_name"
    echo "开始时间: $(date)"
    echo ""
    
    # 记录实验开始时间
    START_TIME=$(date +%s)
    
    # 清理GPU缓存
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    
    # 运行训练
    LOG_FILE="$RESULTS_DIR/${exp_name}.log"
    echo "日志文件: $LOG_FILE"
    echo ""
    
    if python scripts/2_finetune_whisper_lora.py --config "$config_file" 2>&1 | tee "$LOG_FILE"; then
        # 训练成功
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        # 提取结果
        WER=$(grep "eval_wer" "$LOG_FILE" | tail -1 | grep -oP "(?<='eval_wer': )[0-9.]+")
        TRAIN_LOSS=$(grep "train_loss" "$LOG_FILE" | tail -1 | grep -oP "(?<='train_loss': )[0-9.]+")
        
        echo ""
        echo "✅ 实验完成！"
        echo "  WER: ${WER:-N/A}"
        echo "  训练Loss: ${TRAIN_LOSS:-N/A}"
        echo "  耗时: $((DURATION / 60))分$((DURATION % 60))秒"
        echo ""
        
        # 保存结果摘要
        cat >> "$RESULTS_DIR/summary.txt" <<EOF
实验: $exp_name
描述: $desc
配置: $config_file
WER: ${WER:-N/A}
训练Loss: ${TRAIN_LOSS:-N/A}
耗时: $((DURATION / 60))分$((DURATION % 60))秒
完成时间: $(date)
---
EOF
        
    else
        # 训练失败
        echo ""
        echo "❌ 实验失败！"
        echo "  查看日志: $LOG_FILE"
        echo ""
        
        cat >> "$RESULTS_DIR/summary.txt" <<EOF
实验: $exp_name
描述: $desc
状态: 失败
查看日志: $LOG_FILE
---
EOF
    fi
    
    CURRENT=$((CURRENT + 1))
    
    # 实验间隔
    if [ $CURRENT -le $TOTAL_EXPERIMENTS ]; then
        echo "等待5秒后开始下一个实验..."
        sleep 5
    fi
done

echo ""
echo "========================================"
echo "所有实验完成！"
echo "========================================"
echo "结束时间: $(date)"
echo ""
echo "结果摘要："
cat "$RESULTS_DIR/summary.txt"
echo ""
echo "详细结果目录: $RESULTS_DIR"
echo ""
echo "分析结果："
echo "  python scripts/analyze_experiments.py --results_dir $RESULTS_DIR"
echo ""

