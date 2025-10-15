#!/usr/bin/env python3
"""
测试中文ASR模型
"""
import sys
import logging
from pathlib import Path
import yaml

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.models.asr.chinese_asr import create_chinese_asr_model

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """加载模型配置"""
    config_path = Path("config/model_config.yaml")
    
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return None
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config['model']['asr']

def test_asr_model():
    """测试ASR模型"""
    
    logger.info("🚀 开始测试中文ASR模型...")
    
    # 1. 加载配置
    config = load_config()
    if not config:
        return False
    
    logger.info(f"使用模型: {config.get('model_name')}")
    
    # 2. 查找样本音频文件
    sample_dir = Path("data/raw/audio/samples")
    if not sample_dir.exists():
        logger.error("样本目录不存在，请先运行: python scripts/download_seniortalk.py --extract_samples")
        return False
    
    audio_files = list(sample_dir.rglob("*.wav"))
    if not audio_files:
        logger.error("未找到音频文件")
        return False
    
    logger.info(f"找到 {len(audio_files)} 个音频文件")
    
    # 3. 创建ASR模型
    try:
        logger.info("正在加载ASR模型（首次运行会下载模型，请稍等）...")
        asr_model = create_chinese_asr_model(config)
        logger.info("✅ ASR模型加载成功")
    except Exception as e:
        logger.error(f"❌ ASR模型加载失败: {e}")
        return False
    
    # 4. 测试第一个音频文件
    test_audio = audio_files[0]
    logger.info(f"🎵 测试音频文件: {test_audio.name}")
    
    try:
        # 执行ASR
        result = asr_model.forward(str(test_audio))
        
        # 打印结果
        logger.info("📝 ASR识别结果:")
        logger.info(f"  原始文本: {result.text}")
        logger.info(f"  停顿信息:")
        logger.info(f"    停顿次数: {result.pause_info.get('pause_count', 0)}")
        logger.info(f"    停顿比例: {result.pause_info.get('pause_ratio', 0):.2%}")
        logger.info(f"    语速: {result.pause_info.get('speech_rate', 0):.1f} 字/分钟")
        logger.info(f"  音频时长: {result.pause_info.get('audio_duration', 0):.1f} 秒")
        
        if result.segments:
            logger.info(f"  分段数量: {len(result.segments)}")
            logger.info(f"  置信度: {sum(result.confidence_scores)/len(result.confidence_scores):.2f}")
        
        logger.info("✅ ASR测试成功!")
        return True
        
    except Exception as e:
        logger.error(f"❌ ASR处理失败: {e}")
        return False

def main():
    """主函数"""
    
    success = test_asr_model()
    
    if success:
        logger.info("🎉 ASR模型测试通过！")
        logger.info("现在可以运行完整的预处理流程")
    else:
        logger.error("💥 ASR模型测试失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 