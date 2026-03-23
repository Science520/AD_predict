#!/bin/bash
# 快速使用最佳Whisper模型

echo "========================================="
echo "🏆 Whisper-medium 最佳模型使用指南"
echo "========================================="
echo ""

echo "📊 实验结果总结:"
echo "----------------------------------------"
echo "实验1 (高LoRA rank):  WER = 0.9540 ⭐ 最佳"
echo "实验2 (低学习率):     WER = 0.9852"
echo "实验3 (大batch):      WER = 0.9544 ⭐ 最佳"
echo "实验4 (激进训练):     WER = 1.1183 ❌"
echo ""

echo "🎯 推荐模型（二选一）:"
echo "----------------------------------------"
echo "选项1: exp1_high_rank (checkpoint-100)"
echo "  路径: /data/AD_predict/exp1_high_rank/checkpoint-100/"
echo "  WER: 0.9540"
echo "  特点: 最佳准确性"
echo ""
echo "选项2: exp3_large_batch (checkpoint-750)"
echo "  路径: /data/AD_predict/exp3_large_batch/checkpoint-750/"
echo "  WER: 0.9544"
echo "  特点: 训练速度快60%，性能相近"
echo ""

echo "📁 查看详细报告:"
echo "----------------------------------------"
echo "cat /data/AD_predict/all_experiments_20251022_140017/COMPLETE_ANALYSIS.md"
echo ""

echo "⚠️  重要提醒:"
echo "----------------------------------------"
echo "WER = 0.95 意味着95%的词识别错误"
echo "这是因为数据量严重不足（755样本 vs 需要10,000+小时）"
echo "建议: 优先扩大数据集，而非调整模型参数"
echo ""

echo "🔍 使用最佳模型进行推理:"
echo "----------------------------------------"
echo "from transformers import WhisperForConditionalGeneration"
echo "from peft import PeftModel"
echo ""
echo "# 加载模型"
echo "base_model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')"
echo "model = PeftModel.from_pretrained(base_model, '/data/AD_predict/exp1_high_rank/checkpoint-100/')"
echo ""
echo "# 进行推理"
echo "# ... (使用WhisperProcessor处理音频)"
echo ""

# 询问是否需要运行FunASR
echo ""
echo "❓ 是否需要运行FunASR实验？"
echo "----------------------------------------"
read -p "FunASR可能在中文方言上表现更好，但需要安装。是否安装并运行？(y/n): " answer

if [ "$answer" = "y" ]; then
    echo ""
    echo "正在安装FunASR..."
    conda activate graph
    pip install funasr modelscope
    
    echo ""
    echo "开始运行FunASR实验..."
    cd /home/saisai/AD_predict/AD_predict
    bash scripts/run_funasr_experiments.sh
else
    echo ""
    echo "跳过FunASR实验。"
    echo "如果以后需要运行，执行:"
    echo "  pip install funasr modelscope"
    echo "  bash scripts/run_funasr_experiments.sh"
fi

echo ""
echo "========================================="
echo "✅ 完成！"
echo "========================================="

