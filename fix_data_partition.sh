#!/bin/bash
#
# /data 分区只读问题 - 自动修复脚本
# 需要 sudo 权限运行
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "   /data 分区只读问题诊断与修复"
echo "========================================"
echo ""

# 检查是否有 sudo 权限
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}✗ 此脚本需要 sudo 权限${NC}"
    echo "请运行: sudo $0"
    exit 1
fi

DEVICE="/dev/mapper/vg_data-lv_data"
MOUNT_POINT="/data"

echo "Step 1: 诊断当前状态"
echo "----------------------------------------"

# 检查挂载状态
echo "当前挂载状态:"
mount | grep $MOUNT_POINT || echo "  未找到挂载信息"
echo ""

# 检查文件系统状态
echo "文件系统状态:"
tune2fs -l $DEVICE 2>/dev/null | grep -i "state" || echo "  无法读取状态"
echo ""

# 尝试写入测试
TEST_FILE="$MOUNT_POINT/.test_write_$$"
echo "写入测试:"
if touch $TEST_FILE 2>/dev/null; then
    rm -f $TEST_FILE
    echo -e "  ${GREEN}✓ 可以写入，无需修复${NC}"
    exit 0
else
    echo -e "  ${RED}✗ 无法写入，需要修复${NC}"
fi
echo ""

echo "Step 2: 尝试简单修复（重新挂载）"
echo "----------------------------------------"

echo "尝试重新挂载为读写模式..."
if mount -o remount,rw $MOUNT_POINT 2>/dev/null; then
    echo -e "${GREEN}✓ 重新挂载成功${NC}"
    
    # 再次测试写入
    if touch $TEST_FILE 2>/dev/null; then
        rm -f $TEST_FILE
        echo -e "${GREEN}✓ 写入测试通过！问题已解决${NC}"
        echo ""
        echo "修复完成！现在可以正常使用 /data 了"
        exit 0
    else
        echo -e "${YELLOW}⚠ 重新挂载后仍无法写入${NC}"
    fi
else
    echo -e "${YELLOW}⚠ 重新挂载失败${NC}"
fi
echo ""

echo "Step 3: 深度修复（卸载并检查）"
echo "----------------------------------------"
echo -e "${YELLOW}⚠ 警告：此操作会暂时断开对 /data 的访问${NC}"
echo ""

# 检查是否有进程在使用
echo "检查正在使用 /data 的进程:"
USING_PROCS=$(lsof +D $MOUNT_POINT 2>/dev/null | tail -n +2 | wc -l)
if [ "$USING_PROCS" -gt 0 ]; then
    echo -e "${YELLOW}  发现 $USING_PROCS 个进程正在使用 /data${NC}"
    lsof +D $MOUNT_POINT 2>/dev/null | head -n 10
    echo ""
    read -p "是否终止这些进程并继续？(yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
        echo "操作已取消"
        exit 1
    fi
    
    echo "终止进程..."
    fuser -km $MOUNT_POINT 2>/dev/null || true
    sleep 2
else
    echo "  没有进程正在使用 /data"
fi
echo ""

# 卸载
echo "卸载 $MOUNT_POINT..."
if umount $MOUNT_POINT 2>/dev/null; then
    echo -e "${GREEN}✓ 卸载成功${NC}"
else
    echo -e "${YELLOW}尝试强制卸载...${NC}"
    umount -f $MOUNT_POINT 2>/dev/null || umount -l $MOUNT_POINT 2>/dev/null
    sleep 1
fi
echo ""

# 文件系统检查
echo "运行文件系统检查..."
echo -e "${YELLOW}这可能需要几分钟，请耐心等待...${NC}"
echo ""

if e2fsck -f -y $DEVICE; then
    echo -e "${GREEN}✓ 文件系统检查完成${NC}"
else
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 1 ]; then
        echo -e "${GREEN}✓ 文件系统已修复${NC}"
    else
        echo -e "${RED}✗ 文件系统检查失败 (exit code: $EXIT_CODE)${NC}"
    fi
fi
echo ""

# 重新挂载
echo "重新挂载..."
if mount $MOUNT_POINT; then
    echo -e "${GREEN}✓ 挂载成功${NC}"
else
    echo -e "${RED}✗ 挂载失败${NC}"
    exit 1
fi
echo ""

# 最终验证
echo "Step 4: 验证修复结果"
echo "----------------------------------------"

if touch $TEST_FILE 2>/dev/null; then
    rm -f $TEST_FILE
    echo -e "${GREEN}✓ 写入测试通过${NC}"
    echo -e "${GREEN}✓ 问题已成功修复！${NC}"
    echo ""
    echo "现在可以正常使用 /data 分区了"
    
    # 显示状态
    echo ""
    echo "当前状态:"
    df -h $MOUNT_POINT
    echo ""
    mount | grep $MOUNT_POINT
else
    echo -e "${RED}✗ 修复后仍无法写入${NC}"
    echo ""
    echo "建议进一步诊断："
    echo "  1. 检查硬件状态"
    echo "  2. 查看系统日志: dmesg | grep -i error"
    echo "  3. 联系系统管理员"
    exit 1
fi

