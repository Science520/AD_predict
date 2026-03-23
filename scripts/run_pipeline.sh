#!/bin/bash

# Whisper方言微调完整流程脚本
# 依次执行：数据验证 -> 数据预处理 -> 模型微调

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Whisper多方言老年人语音识别微调流程"
echo "=========================================="

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "项目目录: $PROJECT_DIR"
cd "$PROJECT_DIR"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到Python"
    exit 1
fi

echo "✅ Python版本: $(python --version)"

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU信息:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  警告: 未检测到GPU，将使用CPU训练（速度较慢）"
fi

echo ""
echo "=========================================="
echo "步骤 0: 数据验证"
echo "=========================================="
python scripts/0_validate_data.py

# 询问用户是否继续
echo ""
read -p "数据验证完成。是否继续进行数据预处理？ (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "流程已取消"
    exit 0
fi

echo ""
echo "=========================================="
echo "步骤 1: 数据预处理"
echo "=========================================="
python scripts/1_prepare_dataset.py

if [ $? -ne 0 ]; then
    echo "❌ 数据预处理失败"
    exit 1
fi

# 询问用户是否继续
echo ""
read -p "数据预处理完成。是否开始模型微调？ (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "流程已暂停。可以稍后运行："
    echo "  python scripts/2_finetune_whisper_lora.py"
    exit 0
fi

echo ""
echo "=========================================="
echo "步骤 2: Whisper LoRA微调"
echo "=========================================="
python scripts/2_finetune_whisper_lora.py

if [ $? -ne 0 ]; then
    echo "❌ 模型微调失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "🎉 完整流程执行成功！"
echo "=========================================="

echo ""
echo "📁 输出文件："
echo "  - 处理后的数据集: ./processed_data"
echo "  - LoRA适配器: ./whisper_lora_dialect/final_adapter"
echo "  - 训练日志: ./whisper_lora_dialect/logs"
echo ""
echo "🚀 测试模型："
echo "  python scripts/4_inference_test.py --audio YOUR_AUDIO.wav"
echo ""
echo "📖 详细文档："
echo "  查看 WHISPER_FINETUNING_README.md"


