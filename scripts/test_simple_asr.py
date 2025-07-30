#!/usr/bin/env python3
"""
简化版ASR测试 - 直接测试Whisper功能
"""
import sys
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_whisper_installation():
    """测试Whisper是否正确安装"""
    try:
        import whisper
        logger.info("✅ Whisper已安装")
        
        # 列出可用模型
        models = whisper.available_models()
        logger.info(f"可用模型: {models}")
        
        return True
    except ImportError:
        logger.error("❌ 需要安装Whisper: pip install openai-whisper")
        return False

def test_audio_libraries():
    """测试音频处理库"""
    try:
        import librosa
        import soundfile as sf
        logger.info("✅ 音频处理库已安装")
        return True
    except ImportError as e:
        logger.error(f"❌ 音频库缺失: {e}")
        return False

def test_sample_asr():
    """测试ASR功能"""
    
    # 检查样本文件
    sample_dir = Path("data/raw/audio/samples")
    if not sample_dir.exists():
        logger.error("样本目录不存在")
        return False
    
    audio_files = list(sample_dir.rglob("*.wav"))
    if not audio_files:
        logger.error("未找到音频文件")
        return False
    
    test_file = audio_files[0]
    logger.info(f"测试文件: {test_file}")
    
    try:
        import whisper
        import librosa
        
        # 加载小模型进行测试
        logger.info("加载Whisper base模型...")
        model = whisper.load_model("base")
        
        # 转录音频
        logger.info("开始转录...")
        result = model.transcribe(str(test_file), language='zh')
        
        logger.info("🎉 转录成功!")
        logger.info(f"识别文本: {result['text']}")
        
        # 检查是否有分段信息
        if 'segments' in result:
            logger.info(f"分段数量: {len(result['segments'])}")
            for i, seg in enumerate(result['segments'][:3]):  # 只显示前3个
                logger.info(f"  段{i+1}: {seg['text']} ({seg['start']:.1f}s-{seg['end']:.1f}s)")
        
        return True
        
    except Exception as e:
        logger.error(f"ASR测试失败: {e}")
        return False

def main():
    """主函数"""
    
    logger.info("🚀 开始ASR环境测试...")
    
    # 1. 检查Whisper
    if not test_whisper_installation():
        logger.error("请安装: pip install openai-whisper")
        return
    
    # 2. 检查音频库
    if not test_audio_libraries():
        logger.error("请安装: pip install librosa soundfile")
        return
    
    # 3. 测试ASR
    if test_sample_asr():
        logger.info("✅ ASR测试通过！")
        logger.info("环境配置正确，可以继续开发")
    else:
        logger.error("❌ ASR测试失败")

if __name__ == "__main__":
    main() 