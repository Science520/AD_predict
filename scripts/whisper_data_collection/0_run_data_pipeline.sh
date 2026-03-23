#!/bin/bash

# Whisper方言数据收集完整流程
# 步骤：分析采样 -> 选择性下载 -> 爬取字幕 -> 数据预处理

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Whisper方言数据收集完整流程"
echo "=========================================="

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "项目目录: $PROJECT_DIR"
cd "$PROJECT_DIR"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到Python"
    exit 1
fi

echo "✅ Python版本: $(python --version)"

# ==========================================
# 步骤 1: 分析数据分布并生成采样计划
# ==========================================
echo ""
echo "=========================================="
echo "步骤 1: 分析数据分布并生成采样计划"
echo "=========================================="

python scripts/whisper_data_collection/1_analyze_and_sample.py

if [ $? -ne 0 ]; then
    echo "❌ 采样分析失败"
    exit 1
fi

# 询问用户是否继续
echo ""
read -p "采样计划已生成。是否继续下载视频？ (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "流程已暂停。可以稍后运行："
    echo "  bash scripts/whisper_data_collection/0_run_data_pipeline.sh --skip-analysis"
    exit 0
fi

# ==========================================
# 步骤 2: 选择性下载视频
# ==========================================
echo ""
echo "=========================================="
echo "步骤 2: 选择性下载视频和提取音频"
echo "=========================================="

# 检查依赖
if ! command -v you-get &> /dev/null; then
    echo "❌ 未安装 you-get"
    echo "请安装: pip install you-get"
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "❌ 未安装 ffmpeg"
    echo "请安装: apt-get install ffmpeg"
    exit 1
fi

# 询问是否限制下载数量（用于测试）
echo ""
read -p "是否限制下载数量？(y=测试模式下载5个, n=全部下载) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    MAX_DOWNLOADS="--max_downloads 5"
    echo "测试模式：只下载5个视频"
else
    MAX_DOWNLOADS=""
    echo "完整模式：下载所有计划视频"
fi

python scripts/whisper_data_collection/2_selective_download.py $MAX_DOWNLOADS

if [ $? -ne 0 ]; then
    echo "❌ 视频下载失败"
    exit 1
fi

# 询问用户是否继续
echo ""
read -p "视频下载完成。是否继续爬取字幕？ (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "流程已暂停。可以稍后运行："
    echo "  python scripts/whisper_data_collection/3_scrape_subtitles.py"
    exit 0
fi

# ==========================================
# 步骤 3: 爬取字幕并标注
# ==========================================
echo ""
echo "=========================================="
echo "步骤 3: 爬取字幕并生成转录文本"
echo "=========================================="

# 询问Whisper备选策略
echo ""
echo "字幕获取策略："
echo "  1. 优先使用Bilibili字幕"
echo "  2. 如果没有字幕，使用Whisper转录（需要时间）"
echo ""
read -p "是否启用Whisper作为备选？ (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    WHISPER_FLAG=""
    
    # 选择Whisper模型
    echo ""
    echo "选择Whisper模型大小："
    echo "  1. tiny   (最快，质量最低)"
    echo "  2. base   (较快，质量一般) [推荐]"
    echo "  3. small  (中等速度和质量)"
    echo "  4. medium (较慢，质量较好)"
    echo "  5. large  (最慢，质量最好)"
    echo ""
    read -p "请选择 (1-5，默认2): " choice
    
    case $choice in
        1) WHISPER_MODEL="tiny" ;;
        3) WHISPER_MODEL="small" ;;
        4) WHISPER_MODEL="medium" ;;
        5) WHISPER_MODEL="large" ;;
        *) WHISPER_MODEL="base" ;;
    esac
    
    echo "使用Whisper模型: $WHISPER_MODEL"
    WHISPER_FLAG="--whisper_model $WHISPER_MODEL"
else
    WHISPER_FLAG="--no_whisper_fallback"
    echo "仅使用Bilibili字幕"
fi

python scripts/whisper_data_collection/3_scrape_subtitles.py $WHISPER_FLAG

if [ $? -ne 0 ]; then
    echo "❌ 字幕爬取失败"
    exit 1
fi

# ==========================================
# 完成
# ==========================================
echo ""
echo "=========================================="
echo "🎉 数据收集流程完成！"
echo "=========================================="

echo ""
echo "📁 生成的文件："
echo "  - 采样计划: data/sampling_plan.json"
echo "  - 下载结果: data/download_results.json"
echo "  - 转录结果: data/transcript_results.json"
echo "  - 音频文件: data/raw/audio/elderly_audios/"
echo "  - 转录文本: data/raw/audio/result/"
echo ""
echo "📊 数据统计："

if [ -f "data/download_results.json" ]; then
    SUCCESS_COUNT=$(python -c "import json; data=json.load(open('data/download_results.json')); print(len(data['success']))")
    echo "  成功下载视频: $SUCCESS_COUNT 个"
fi

if [ -f "data/transcript_results.json" ]; then
    BILIBILI_COUNT=$(python -c "import json; data=json.load(open('data/transcript_results.json')); print(len(data['bilibili_subtitle']))")
    WHISPER_COUNT=$(python -c "import json; data=json.load(open('data/transcript_results.json')); print(len(data['whisper_transcribed']))")
    echo "  Bilibili字幕: $BILIBILI_COUNT 个"
    echo "  Whisper转录: $WHISPER_COUNT 个"
fi

echo ""
echo "🚀 下一步："
echo "  1. 检查数据质量:"
echo "     python scripts/0_validate_data.py"
echo ""
echo "  2. 运行数据预处理:"
echo "     python scripts/1_prepare_dataset.py"
echo ""
echo "  3. 开始模型微调:"
echo "     python scripts/2_finetune_whisper_lora.py"

