#!/bin/bash
# 🌙 完整的8小时自动化实验框架
# Whisper (4组) + FunASR (4组) = 8组参数对比实验
# 特性：失败不停止，自动继续下一个实验

set +e  # ⚠️ 不在错误时退出，继续执行

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
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 创建结果目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/data/AD_predict/all_experiments_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "🌙 8小时全自动实验框架"
echo "========================================"
echo "开始时间: $(date)"
echo "结果目录: $RESULTS_DIR"
echo "实验总数: 8组 (4个Whisper + 4个FunASR)"
echo ""

# 记录开始时间
GLOBAL_START=$(date +%s)

# ========================================
# 第一阶段: Whisper实验 (4组)
# ========================================

echo ""
echo "========================================"
echo "📊 第一阶段: Whisper参数对比实验"
echo "========================================"
echo ""

WHISPER_EXPERIMENTS=(
    "whisper_exp1:configs/exp1_high_rank.yaml:Whisper-高LoRA_rank"
    "whisper_exp2:configs/exp2_low_lr.yaml:Whisper-低学习率"
    "whisper_exp3:configs/exp3_large_batch.yaml:Whisper-大batch"
    "whisper_exp4:configs/exp4_aggressive.yaml:Whisper-激进训练"
)

WHISPER_SUCCESS=0
WHISPER_FAIL=0

for exp in "${WHISPER_EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name config_file desc <<< "$exp"
    
    echo ""
    echo "========================================" | tee -a "$RESULTS_DIR/summary.txt"
    echo "🔬 $desc" | tee -a "$RESULTS_DIR/summary.txt"
    echo "========================================" | tee -a "$RESULTS_DIR/summary.txt"
    echo "配置: $config_file"
    echo "输出: /data/AD_predict/$exp_name"
    echo "开始: $(date)"
    echo ""
    
    EXP_START=$(date +%s)
    LOG_FILE="$RESULTS_DIR/${exp_name}.log"
    
    # 清理GPU缓存
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    
    # 运行实验
    if python scripts/2_finetune_whisper_lora.py --config "$config_file" 2>&1 | tee "$LOG_FILE"; then
        EXP_END=$(date +%s)
        DURATION=$((EXP_END - EXP_START))
        
        # 提取结果
        WER=$(grep "eval_wer" "$LOG_FILE" | tail -1 | grep -oP "(?<='eval_wer': )[0-9.]+")
        LOSS=$(grep "train_loss" "$LOG_FILE" | tail -1 | grep -oP "(?<='train_loss': )[0-9.]+")
        
        echo ""
        echo "✅ 实验成功！" | tee -a "$RESULTS_DIR/summary.txt"
        echo "  WER: ${WER:-N/A}" | tee -a "$RESULTS_DIR/summary.txt"
        echo "  Loss: ${LOSS:-N/A}" | tee -a "$RESULTS_DIR/summary.txt"
        echo "  耗时: $((DURATION / 60))分$((DURATION % 60))秒" | tee -a "$RESULTS_DIR/summary.txt"
        
        WHISPER_SUCCESS=$((WHISPER_SUCCESS + 1))
        
        cat >> "$RESULTS_DIR/results.csv" <<EOF
$exp_name,$desc,${WER:-N/A},${LOSS:-N/A},$((DURATION / 60))
EOF
    else
        echo ""
        echo "❌ 实验失败！" | tee -a "$RESULTS_DIR/summary.txt"
        echo "  查看日志: $LOG_FILE" | tee -a "$RESULTS_DIR/summary.txt"
        echo "  ⏭️  继续下一个实验..." | tee -a "$RESULTS_DIR/summary.txt"
        
        WHISPER_FAIL=$((WHISPER_FAIL + 1))
    fi
    
    echo "---" >> "$RESULTS_DIR/summary.txt"
    sleep 5
done

echo ""
echo "========================================"
echo "📊 Whisper实验完成"
echo "========================================"
echo "成功: $WHISPER_SUCCESS / 失败: $WHISPER_FAIL"
echo ""

# ========================================
# 第二阶段: FunASR实验 (4组)
# ========================================

echo ""
echo "========================================"
echo "📊 第二阶段: FunASR参数对比实验"
echo "========================================"
echo ""

FUNASR_EXPERIMENTS=(
    "funasr_exp1:configs/funasr_exp1_high_rank.yaml:FunASR-高LoRA_rank"
    "funasr_exp2:configs/funasr_exp2_low_lr.yaml:FunASR-低学习率"
    "funasr_exp3:configs/funasr_exp3_large_batch.yaml:FunASR-大batch"
    "funasr_exp4:configs/funasr_exp4_baseline.yaml:FunASR-baseline"
)

FUNASR_SUCCESS=0
FUNASR_FAIL=0

for exp in "${FUNASR_EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name config_file desc <<< "$exp"
    
    echo ""
    echo "========================================" | tee -a "$RESULTS_DIR/summary.txt"
    echo "🔬 $desc" | tee -a "$RESULTS_DIR/summary.txt"
    echo "========================================" | tee -a "$RESULTS_DIR/summary.txt"
    echo "配置: $config_file"
    echo "输出: /data/AD_predict/$exp_name"
    echo "开始: $(date)"
    echo ""
    
    EXP_START=$(date +%s)
    LOG_FILE="$RESULTS_DIR/${exp_name}.log"
    
    # 清理GPU缓存
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    
    # 检查FunASR脚本是否存在且可执行
    if [ -f "scripts/2_finetune_funasr_lora_FunASR.py" ]; then
        # 运行FunASR实验
        if python scripts/2_finetune_funasr_lora_FunASR.py --config "$config_file" 2>&1 | tee "$LOG_FILE"; then
            EXP_END=$(date +%s)
            DURATION=$((EXP_END - EXP_START))
            
            # 提取结果
            WER=$(grep "eval_wer\|eval_cer" "$LOG_FILE" | tail -1 | grep -oP "[0-9.]+")
            LOSS=$(grep "train_loss" "$LOG_FILE" | tail -1 | grep -oP "(?<='train_loss': )[0-9.]+")
            
            echo ""
            echo "✅ 实验成功！" | tee -a "$RESULTS_DIR/summary.txt"
            echo "  WER/CER: ${WER:-N/A}" | tee -a "$RESULTS_DIR/summary.txt"
            echo "  Loss: ${LOSS:-N/A}" | tee -a "$RESULTS_DIR/summary.txt"
            echo "  耗时: $((DURATION / 60))分$((DURATION % 60))秒" | tee -a "$RESULTS_DIR/summary.txt"
            
            FUNASR_SUCCESS=$((FUNASR_SUCCESS + 1))
            
            cat >> "$RESULTS_DIR/results.csv" <<EOF
$exp_name,$desc,${WER:-N/A},${LOSS:-N/A},$((DURATION / 60))
EOF
        else
            echo ""
            echo "❌ 实验失败！" | tee -a "$RESULTS_DIR/summary.txt"
            echo "  查看日志: $LOG_FILE" | tee -a "$RESULTS_DIR/summary.txt"
            echo "  ⏭️  继续下一个实验..." | tee -a "$RESULTS_DIR/summary.txt"
            
            FUNASR_FAIL=$((FUNASR_FAIL + 1))
        fi
    else
        echo "⚠️  FunASR脚本未实现，跳过" | tee -a "$RESULTS_DIR/summary.txt"
        echo "  创建占位符以便未来实现" | tee -a "$RESULTS_DIR/summary.txt"
        FUNASR_FAIL=$((FUNASR_FAIL + 1))
    fi
    
    echo "---" >> "$RESULTS_DIR/summary.txt"
    sleep 5
done

echo ""
echo "========================================"
echo "📊 FunASR实验完成"
echo "========================================"
echo "成功: $FUNASR_SUCCESS / 失败: $FUNASR_FAIL"
echo ""

# ========================================
# 生成最终报告
# ========================================

GLOBAL_END=$(date +%s)
TOTAL_DURATION=$((GLOBAL_END - GLOBAL_START))

echo ""
echo "========================================"
echo "🎉 所有实验完成！"
echo "========================================"
echo "总耗时: $((TOTAL_DURATION / 3600))小时$((TOTAL_DURATION % 3600 / 60))分"
echo ""
echo "📊 实验统计:"
echo "  Whisper: $WHISPER_SUCCESS 成功 / $WHISPER_FAIL 失败"
echo "  FunASR:  $FUNASR_SUCCESS 成功 / $FUNASR_FAIL 失败"
echo "  总计:    $((WHISPER_SUCCESS + FUNASR_SUCCESS)) 成功 / $((WHISPER_FAIL + FUNASR_FAIL)) 失败"
echo ""
echo "📁 结果目录: $RESULTS_DIR"
echo ""

# 生成CSV表头
if [ -f "$RESULTS_DIR/results.csv" ]; then
    sed -i '1i实验名称,描述,WER,训练Loss,耗时(分钟)' "$RESULTS_DIR/results.csv"
fi

# 生成最终报告
cat > "$RESULTS_DIR/FINAL_REPORT.md" <<EOF
# 🎉 Whisper + FunASR 参数对比实验报告

**生成时间**: $(date)  
**总耗时**: $((TOTAL_DURATION / 3600))小时$((TOTAL_DURATION % 3600 / 60))分  
**实验数量**: 8组  

---

## 📊 实验统计

| 模型 | 成功 | 失败 | 成功率 |
|------|------|------|--------|
| **Whisper** | $WHISPER_SUCCESS | $WHISPER_FAIL | $((WHISPER_SUCCESS * 100 / 4))% |
| **FunASR** | $FUNASR_SUCCESS | $FUNASR_FAIL | $((FUNASR_SUCCESS * 100 / 4))% |
| **总计** | $((WHISPER_SUCCESS + FUNASR_SUCCESS)) | $((WHISPER_FAIL + FUNASR_FAIL)) | $((((WHISPER_SUCCESS + FUNASR_SUCCESS) * 100) / 8))% |

---

## 📋 详细结果

EOF

# 如果有CSV结果，添加到报告
if [ -f "$RESULTS_DIR/results.csv" ]; then
    echo '```' >> "$RESULTS_DIR/FINAL_REPORT.md"
    cat "$RESULTS_DIR/results.csv" >> "$RESULTS_DIR/FINAL_REPORT.md"
    echo '```' >> "$RESULTS_DIR/FINAL_REPORT.md"
    echo "" >> "$RESULTS_DIR/FINAL_REPORT.md"
fi

cat >> "$RESULTS_DIR/FINAL_REPORT.md" <<EOF

## 📂 文件位置

- **实验日志**: \`$RESULTS_DIR/*.log\`
- **模型输出**: \`/data/AD_predict/whisper_exp*/\`
- **模型输出**: \`/data/AD_predict/funasr_exp*/\`
- **TensorBoard**: \`/data/AD_predict/logs_*\`

---

## 🔍 查看结果

\`\`\`bash
# 查看完整报告
cat $RESULTS_DIR/FINAL_REPORT.md

# 查看摘要
cat $RESULTS_DIR/summary.txt

# 查看某个实验日志
tail -100 $RESULTS_DIR/whisper_exp1.log
\`\`\`

---

## 📈 后续步骤

1. 分析WER最低的配置
2. 选择最优模型进行最终训练
3. 生成月报对比材料

EOF

echo "✅ 最终报告已生成: $RESULTS_DIR/FINAL_REPORT.md"
echo ""
echo "🎯 快速查看结果:"
echo "  cat $RESULTS_DIR/FINAL_REPORT.md"
echo ""
echo "📊 TensorBoard可视化:"
echo "  tensorboard --logdir=/data/AD_predict/logs_exp1,/data/AD_predict/logs_funasr_exp1"
echo ""

# 发送完成通知（如果有mail命令）
if command -v mail &> /dev/null; then
    echo "实验完成！查看 $RESULTS_DIR/FINAL_REPORT.md" | mail -s "ASR实验完成" $(whoami)
fi

echo "🎉 全部完成！祝好梦！"

