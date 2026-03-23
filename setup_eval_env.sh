#!/bin/bash
# SeniorTalk 评估环境设置脚本

set -e

echo "=========================================="
echo "设置 SeniorTalk 评估环境"
echo "=========================================="
echo

# 激活 conda 环境
echo "1. 激活 conda 环境 'graph'..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graph
echo "   ✓ 当前环境: $CONDA_DEFAULT_ENV"
echo

# 检查并卸载问题包
echo "2. 检查并移除问题包..."
if pip show torchcodec &>/dev/null; then
    echo "   发现 torchcodec，正在卸载..."
    pip uninstall -y torchcodec
    echo "   ✓ torchcodec 已卸载"
else
    echo "   ✓ torchcodec 未安装（正确）"
fi
echo

# 安装/更新依赖
echo "3. 安装评估所需依赖..."
pip install -r requirements_eval.txt
echo "   ✓ 依赖安装完成"
echo

# 验证关键包
echo "4. 验证关键包..."
python -c "
import sys
packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'peft': 'PEFT',
    'datasets': 'Datasets',
    'soundfile': 'SoundFile',
    'librosa': 'Librosa',
    'jiwer': 'Jiwer',
    'pypinyin': 'Pypinyin'
}

all_ok = True
for pkg, name in packages.items():
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'   ✓ {name}: {version}')
    except ImportError as e:
        print(f'   ✗ {name}: 未安装')
        all_ok = False

if not all_ok:
    print('\\n   ⚠ 部分包未安装，请检查')
    sys.exit(1)
"
echo

# 设置环境变量
echo "5. 配置环境变量..."
export HF_HOME=/tmp/saisai_hf_cache
export TRANSFORMERS_CACHE=/tmp/saisai_hf_cache/transformers
export HF_DATASETS_CACHE=/tmp/saisai_hf_cache/datasets
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p $HF_HOME/transformers $HF_HOME/datasets
echo "   ✓ HF 缓存: $HF_HOME"
echo "   ✓ HF 镜像: $HF_ENDPOINT"
echo

# 验证数据访问
echo "6. 验证数据访问..."
if [ -f "data/raw/seniortalk_full/sentence_data/test-00000-of-00003.parquet" ]; then
    echo "   ✓ 数据文件可访问"
else
    echo "   ✗ 数据文件不可访问"
    echo "   请检查软链接: ls -l data"
    exit 1
fi
echo

echo "=========================================="
echo "✓ 环境设置完成！"
echo "=========================================="
echo
echo "现在可以运行评估："
echo "  ./run_eval_local.sh"
echo
echo "或手动运行："
echo "  python scripts/eval_seniortalk_available_models.py"
echo

