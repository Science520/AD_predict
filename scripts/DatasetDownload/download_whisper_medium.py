"""
下载Whisper-medium模型到本地缓存
"""
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# 设置Hugging Face环境变量
os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 不设置离线模式，允许下载
if 'HF_HUB_OFFLINE' in os.environ:
    del os.environ['HF_HUB_OFFLINE']

print("=" * 80)
print("下载 Whisper-medium 模型")
print("=" * 80)
print("模型: openai/whisper-medium")
print("缓存目录: ~/.cache/huggingface")
print("\n开始下载...")

try:
    # 下载模型
    print("\n1. 下载模型权重...")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-medium",
        cache_dir=os.path.expanduser('~/.cache/huggingface')
    )
    print("✓ 模型权重下载完成")
    
    # 下载processor
    print("\n2. 下载处理器...")
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-medium",
        cache_dir=os.path.expanduser('~/.cache/huggingface')
    )
    print("✓ 处理器下载完成")
    
    print("\n" + "=" * 80)
    print("✅ Whisper-medium 下载成功！")
    print("=" * 80)
    print(f"\n模型信息:")
    print(f"  - 参数量: ~769M")
    print(f"  - 编码器层数: 24")
    print(f"  - 解码器层数: 24")
    print(f"  - 隐藏层维度: 1024")
    print(f"\n现在可以运行训练脚本了:")
    print(f"  python scripts/2_finetune_whisper_lora.py --config configs/training_args_medium.yaml")
    
except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    print("\n可能的解决方案:")
    print("1. 检查网络连接")
    print("2. 尝试使用VPN")
    print("3. 或者手动下载模型文件")
    exit(1)

