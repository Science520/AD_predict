#!/bin/bash
#
# 从本地缓存复制 Whisper 模型到 HF 缓存
#

set -e

echo "========================================"
echo "  复制本地 Whisper 模型"
echo "========================================"
echo ""

# 目标目录（用户的 HF 缓存）
HF_CACHE="$HOME/.cache/huggingface/hub"
mkdir -p "$HF_CACHE"

echo "目标缓存目录: $HF_CACHE"
echo ""

# 源目录
SOURCE_DIR="/data/saisai/cache/whisper"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "✗ 源目录不存在: $SOURCE_DIR"
    echo ""
    echo "尝试其他位置..."
    
    # 尝试其他可能的位置
    ALTERNATIVES=(
        "/data/cache/whisper"
        "/data/saisai/whisper"
        "/mnt/backup/data_backup/saisai/cache/whisper"
    )
    
    for alt in "${ALTERNATIVES[@]}"; do
        if [ -d "$alt" ]; then
            SOURCE_DIR="$alt"
            echo "✓ 找到: $SOURCE_DIR"
            break
        fi
    done
    
    if [ ! -d "$SOURCE_DIR" ]; then
        echo "✗ 无法找到 whisper 模型目录"
        exit 1
    fi
fi

echo "源目录: $SOURCE_DIR"
echo "大小: $(sudo du -sh $SOURCE_DIR 2>/dev/null | cut -f1)"
echo ""

# 检查是否需要 sudo
if [ ! -r "$SOURCE_DIR" ]; then
    echo "需要 sudo 权限来访问源目录..."
    USE_SUDO="sudo"
else
    USE_SUDO=""
fi

echo "开始复制..."
echo "这可能需要几分钟..."
echo ""

# 复制整个目录
if $USE_SUDO rsync -av --progress "$SOURCE_DIR/" "$HF_CACHE/models--openai--whisper-medium/"; then
    echo ""
    echo "✓ 复制成功！"
    echo ""
    echo "验证:"
    ls -lh "$HF_CACHE/models--openai--whisper-medium/" | head -10
    echo ""
    echo "模型已准备好，可以离线使用！"
else
    echo ""
    echo "✗ 复制失败"
    echo ""
    echo "尝试手动复制:"
    echo "  sudo cp -r $SOURCE_DIR $HF_CACHE/models--openai--whisper-medium/"
    exit 1
fi

echo ""
echo "========================================"
echo "  完成！"
echo "========================================"





