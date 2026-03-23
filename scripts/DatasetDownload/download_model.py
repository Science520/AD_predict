#!/usr/bin/env python3
"""
下载Hugging Face模型到本地缓存
"""
import os
import sys
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# 设置Hugging Face环境
os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用镜像加速
# 暂时不设置离线模式，以便下载
if 'HF_HUB_OFFLINE' in os.environ:
    del os.environ['HF_HUB_OFFLINE']

def download_model(model_name):
    """下载指定的Whisper模型"""
    print(f"================================================================================")
    print(f"开始下载模型: {model_name}")
    print(f"================================================================================")
    print(f"缓存目录: {os.environ['HF_HOME']}")
    print(f"镜像站点: {os.environ.get('HF_ENDPOINT', '官方站点')}")
    print()
    
    try:
        print("步骤 1: 下载模型...")
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        print(f"✓ 模型下载成功")
        print(f"  参数数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        
        print("\n步骤 2: 下载处理器...")
        processor = WhisperProcessor.from_pretrained(model_name, language="zh", task="transcribe")
        print(f"✓ 处理器下载成功")
        
        print(f"\n{'='*80}")
        print(f"✅ 模型 {model_name} 下载完成！")
        print(f"{'='*80}")
        print(f"\n现在可以在离线模式下使用该模型")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ 下载失败: {e}")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "openai/whisper-medium"
    
    success = download_model(model_name)
    sys.exit(0 if success else 1)

