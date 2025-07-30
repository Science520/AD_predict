#!/usr/bin/env python3
"""
增强版ASR测试 - 基于成功的简单版本，添加停顿检测和语言特征分析
"""
import sys
import logging
from pathlib import Path
import whisper
import librosa
import numpy as np
import re
from typing import Dict, List

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedChineseASR:
    """增强版中文ASR - 包含停顿检测和语言特征分析"""
    
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)
        self.sample_rate = 16000
        
    def detect_pauses(self, audio_path: str, segments: List[Dict]) -> Dict:
        """检测停顿信息"""
        
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        audio_duration = len(audio) / sr
        
        # 计算音频能量
        frame_length = int(0.025 * sr)  # 25ms
        hop_length = int(0.01 * sr)     # 10ms
        
        energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # 检测静音段
        silence_threshold = np.mean(energy) * 0.1
        silence_mask = energy < silence_threshold
        
        # 转换为时间戳
        time_frames = librosa.frames_to_time(
            np.arange(len(energy)),
            sr=sr,
            hop_length=hop_length
        )
        
        # 找到停顿段落
        pauses = []
        in_pause = False
        pause_start = 0
        min_pause_duration = 0.3  # 最小停顿时长
        
        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_pause:
                pause_start = time_frames[i]
                in_pause = True
            elif not is_silent and in_pause:
                pause_duration = time_frames[i] - pause_start
                if pause_duration >= min_pause_duration:
                    pauses.append({
                        'start': pause_start,
                        'end': time_frames[i],
                        'duration': pause_duration
                    })
                in_pause = False
        
        # 计算停顿统计
        total_pause_time = sum(p['duration'] for p in pauses)
        pause_ratio = total_pause_time / audio_duration if audio_duration > 0 else 0
        
        return {
            'pauses': pauses,
            'total_pause_time': total_pause_time,
            'pause_ratio': pause_ratio,
            'pause_count': len(pauses),
            'audio_duration': audio_duration,
            'average_pause_duration': total_pause_time / len(pauses) if pauses else 0
        }
    
    def add_pause_markers(self, text: str, segments: List[Dict], pause_info: Dict) -> str:
        """在文本中添加停顿标记"""
        
        if not segments:
            return text
            
        pauses = pause_info['pauses']
        result_text = ""
        last_end = 0
        
        for segment in segments:
            seg_start = segment['start']
            seg_text = segment['text']
            
            # 检查是否在此分段前有停顿
            for pause in pauses:
                if last_end <= pause['start'] <= seg_start:
                    if pause['duration'] >= 0.5:  # 只标记较长的停顿
                        result_text += f" <pause:{pause['duration']:.1f}s> "
                        
            result_text += seg_text
            last_end = segment['end']
            
        return result_text.strip()
    
    def calculate_speech_rate(self, text: str, pause_info: Dict) -> float:
        """计算语速 (字/分钟)"""
        
        # 简单的中文字符计数
        char_count = len([c for c in text if c.strip() and not c.isspace() and not '<' in c])
        
        # 减去停顿时间
        effective_duration = pause_info['audio_duration'] - pause_info['total_pause_time']
        
        if effective_duration > 0:
            speech_rate = (char_count / effective_duration) * 60  # 字/分钟
        else:
            speech_rate = 0
            
        return speech_rate
    
    def analyze_language_features(self, text: str) -> Dict:
        """分析语言特征"""
        
        # 基本统计
        char_count = len([c for c in text if c.strip() and not '<' in c])
        
        # 停顿标记统计
        pause_markers = re.findall(r'<pause:(\d+\.?\d*)s>', text)
        pause_count = len(pause_markers)
        
        # 重复检测（简单版）
        words = text.replace('<pause:', '').replace('s>', '').split()
        word_counts = {}
        for word in words:
            if len(word) > 1:  # 只统计长度大于1的词
                word_counts[word] = word_counts.get(word, 0) + 1
        
        repeated_words = {word: count for word, count in word_counts.items() if count > 1}
        
        return {
            'char_count': char_count,
            'pause_markers_count': pause_count,
            'repeated_words': repeated_words,
            'repetition_ratio': len(repeated_words) / len(set(words)) if words else 0,
            'total_words': len(words)
        }
    
    def process_audio(self, audio_path: str) -> Dict:
        """完整的音频处理流程"""
        
        logger.info(f"处理音频文件: {Path(audio_path).name}")
        
        # 1. ASR转录
        result = self.model.transcribe(
            audio_path,
            language='zh',
            word_timestamps=True,
            verbose=False
        )
        
        text = result['text']
        segments = result.get('segments', [])
        
        # 2. 停顿检测
        pause_info = self.detect_pauses(audio_path, segments)
        
        # 3. 添加停顿标记
        text_with_pauses = self.add_pause_markers(text, segments, pause_info)
        
        # 4. 计算语速
        speech_rate = self.calculate_speech_rate(text, pause_info)
        
        # 5. 语言特征分析
        language_features = self.analyze_language_features(text_with_pauses)
        
        return {
            'original_text': text,
            'text_with_pauses': text_with_pauses,
            'segments': segments,
            'pause_info': pause_info,
            'speech_rate': speech_rate,
            'language_features': language_features
        }

def test_enhanced_asr():
    """测试增强版ASR"""
    
    logger.info("🚀 开始增强版ASR测试...")
    
    # 检查样本文件
    sample_dir = Path("data/raw/audio/samples")
    audio_files = list(sample_dir.rglob("*.wav"))
    
    if not audio_files:
        logger.error("未找到音频文件")
        return False
    
    # 创建增强ASR实例
    logger.info("初始化增强版ASR...")
    asr = EnhancedChineseASR()
    
    # 测试多个文件
    for i, audio_file in enumerate(audio_files[:3]):  # 测试前3个文件
        logger.info(f"\n{'='*50}")
        logger.info(f"测试文件 {i+1}/{min(3, len(audio_files))}")
        
        try:
            result = asr.process_audio(str(audio_file))
            
            # 打印结果
            logger.info("📝 处理结果:")
            logger.info(f"  原始文本: {result['original_text']}")
            logger.info(f"  带停顿文本: {result['text_with_pauses']}")
            
            logger.info("⏱️  时间特征:")
            logger.info(f"  音频时长: {result['pause_info']['audio_duration']:.1f}秒")
            logger.info(f"  停顿次数: {result['pause_info']['pause_count']}")
            logger.info(f"  停顿比例: {result['pause_info']['pause_ratio']:.1%}")
            logger.info(f"  语速: {result['speech_rate']:.1f} 字/分钟")
            
            logger.info("🔤 语言特征:")
            lang_features = result['language_features']
            logger.info(f"  字符数: {lang_features['char_count']}")
            logger.info(f"  词汇数: {lang_features['total_words']}")
            if lang_features['repeated_words']:
                logger.info(f"  重复词汇: {lang_features['repeated_words']}")
            
            logger.info("✅ 处理成功!")
            
        except Exception as e:
            logger.error(f"❌ 处理失败: {e}")
            return False
    
    return True

def main():
    """主函数"""
    
    if test_enhanced_asr():
        logger.info("\n🎉 增强版ASR测试全部通过！")
        logger.info("系统已准备好处理中文老年人语音数据")
        logger.info("功能包括：")
        logger.info("  ✓ 中文语音识别") 
        logger.info("  ✓ 停顿检测与标记")
        logger.info("  ✓ 语速计算")
        logger.info("  ✓ 语言特征分析")
        logger.info("  ✓ 重复模式检测")
    else:
        logger.error("❌ 增强版ASR测试失败")

if __name__ == "__main__":
    main() 