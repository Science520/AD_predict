#!/bin/bash
#
# 查找 Backup 中的模型文件
#

echo "========================================"
echo "  寻找 Backup 中的模型文件"
echo "========================================"
echo ""

echo "正在搜索 backup 目录..."
echo ""

# 搜索常见位置
LOCATIONS=(
    "/data/backup"
    "/data/AD_predict/backup"
    "/data/cache/backup"
    "$HOME/backup"
    "$HOME/AD_predict/backup"
    "/mnt/backup"
    "/media/backup"
)

echo "1. 检查常见位置:"
echo "----------------------------------------"
for loc in "${LOCATIONS[@]}"; do
    if [ -d "$loc" ]; then
        echo "✓ 找到: $loc"
        echo "  大小: $(du -sh $loc 2>/dev/null | cut -f1)"
        echo "  内容:"
        ls -lh "$loc" 2>/dev/null | head -10
        echo ""
    fi
done

echo ""
echo "2. 搜索 whisper 模型目录:"
echo "----------------------------------------"
find /data -maxdepth 4 -name "*whisper*" -type d 2>/dev/null | while read dir; do
    echo "✓ 找到: $dir"
    echo "  大小: $(du -sh "$dir" 2>/dev/null | cut -f1)"
    if [ -f "$dir/config.json" ]; then
        echo "  ⭐ 包含 config.json (可能是模型)"
    fi
    echo ""
done

find $HOME -maxdepth 4 -name "*whisper*" -type d 2>/dev/null | while read dir; do
    echo "✓ 找到: $dir"
    echo "  大小: $(du -sh "$dir" 2>/dev/null | cut -f1)"
    if [ -f "$dir/config.json" ]; then
        echo "  ⭐ 包含 config.json (可能是模型)"
    fi
    echo ""
done

echo ""
echo "3. 搜索 backup 目录:"
echo "----------------------------------------"
find /data -maxdepth 3 -name "backup" -type d 2>/dev/null | while read dir; do
    echo "✓ 找到: $dir"
    echo "  大小: $(du -sh "$dir" 2>/dev/null | cut -f1)"
    echo "  内容:"
    ls -lh "$dir" 2>/dev/null | head -10
    echo ""
done

find $HOME -maxdepth 3 -name "backup" -type d 2>/dev/null | while read dir; do
    echo "✓ 找到: $dir"
    echo "  大小: $(du -sh "$dir" 2>/dev/null | cut -f1)"
    echo "  内容:"
    ls -lh "$dir" 2>/dev/null | head -10
    echo ""
done

echo ""
echo "4. 检查现有的 HF 缓存:"
echo "----------------------------------------"
if [ -d "$HOME/.cache/huggingface" ]; then
    echo "✓ HF 缓存目录存在"
    echo "  位置: $HOME/.cache/huggingface"
    echo "  大小: $(du -sh $HOME/.cache/huggingface 2>/dev/null | cut -f1)"
    echo ""
    echo "  已缓存的模型:"
    ls -lh $HOME/.cache/huggingface/hub/ 2>/dev/null | grep "models--" | head -10
else
    echo "✗ HF 缓存目录不存在"
fi

echo ""
echo "========================================"
echo "  搜索完成！"
echo "========================================"
echo ""
echo "如果找到了 backup 或 whisper 目录，请记下完整路径。"
echo "然后告诉我，我会帮你复制到正确位置。"
echo ""

