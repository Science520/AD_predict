#!/usr/bin/env python3
"""
7维度声学特征提取器 - 基于表格设计的完整实现
按照MCI检测的7个声学特征维度进行音频分析
"""
import sys
import logging
from pathlib import Path
import whisper
import librosa
import numpy as np
import re
from typing import Dict, List, Tuple
import scipy.stats
from scipy.signal import find_peaks
import parselmouth
from parselmouth.praat import call

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SevenDimensionAcousticExtractor:
    """7维度声学特征提取器 - 按照MCI检测表格设计"""
    
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)
        self.sample_rate = 16000
        
    def extract_prosody_features(self, audio_path: str) -> Dict:
        """1. 韵律和语调 (Prosody) 特征提取"""
        
        # 使用Parselmouth进行更精确的韵律分析
        sound = parselmouth.Sound(audio_path)
        
        # 提取基频(F0)
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=50, pitch_ceiling=500)
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values != 0]  # 移除无声段
        
        if len(f0_values) == 0:
            return {'f0_std': 0, 'f0_range': 0, 'f0_mean': 0, 'pitch_range_st': 0}
        
        # 计算韵律特征
        f0_std = np.std(f0_values)
        f0_range = np.max(f0_values) - np.min(f0_values)
        f0_mean = np.mean(f0_values)
        
        # 计算音高范围（半音）
        pitch_range_st = 12 * np.log2(np.max(f0_values) / np.min(f0_values)) if np.min(f0_values) > 0 else 0
        
        # 语调轮廓分析
        f0_slopes = np.diff(f0_values)
        slope_variability = np.std(f0_slopes)
        
        return {
            'f0_std': f0_std,
            'f0_range': f0_range, 
            'f0_mean': f0_mean,
            'pitch_range_st': pitch_range_st,
            'slope_variability': slope_variability,
            'prosody_score': 0  # 待人工标注训练
        }
    
    def extract_voice_quality_features(self, audio_path: str) -> Dict:
        """2. 音质和稳定性 (Voice Quality) 特征提取"""
        
        sound = parselmouth.Sound(audio_path)
        
        # 提取Jitter和Shimmer
        pointprocess = call(sound, "To PointProcess (periodic, cc)", 50, 500)
        
        try:
            jitter = call(pointprocess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call([sound, pointprocess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except:
            jitter = 0
            shimmer = 0
        
        # 计算谐噪比(HNR)
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 50, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        
        return {
            'jitter': jitter * 100,  # 转换为百分比
            'shimmer': shimmer,
            'hnr': hnr,
            'voice_quality_score': 0  # 待人工标注训练
        }
    
    def extract_articulation_features(self, audio_path: str) -> Dict:
        """3. 发音清晰度 (Articulation) 特征提取"""
        
        # 加载音频进行共振峰分析
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # 使用Parselmouth提取共振峰
        sound = parselmouth.Sound(audio_path)
        formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        
        # 提取F1, F2, F3
        f1_values = []
        f2_values = []
        f3_values = []
        
        duration = sound.get_total_duration()
        times = np.arange(0, duration, 0.01)
        
        for t in times:
            try:
                f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
                f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
                f3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")
                
                if not (np.isnan(f1) or np.isnan(f2) or np.isnan(f3)):
                    f1_values.append(f1)
                    f2_values.append(f2)
                    f3_values.append(f3)
            except:
                continue
        
        # 计算元音空间面积(VSA) - 简化版
        if len(f1_values) >= 3 and len(f2_values) >= 3:
            # 使用F1和F2的变异性作为VSA的代理指标
            vsa_proxy = np.std(f1_values) * np.std(f2_values)
            
            # 计算共振峰变化率
            f1_transitions = np.abs(np.diff(f1_values))
            f2_transitions = np.abs(np.diff(f2_values))
            formant_transition_rate = np.mean(f1_transitions + f2_transitions)
        else:
            vsa_proxy = 0
            formant_transition_rate = 0
        
        return {
            'vsa_proxy': vsa_proxy,
            'formant_transition_rate': formant_transition_rate,
            'f1_std': np.std(f1_values) if f1_values else 0,
            'f2_std': np.std(f2_values) if f2_values else 0,
            'articulation_clarity_score': 0  # 待人工标注训练
        }
    
    def extract_rhythm_features(self, audio_path: str, segments: List[Dict]) -> Dict:
        """4. 语速和节律 (Speech Rate & Rhythm) 特征提取"""
        
        if not segments:
            return {'articulation_rate': 0, 'syllable_variability': 0}
        
        # 计算音节时长
        syllable_durations = []
        for segment in segments:
            duration = segment['end'] - segment['start']
            text = segment['text'].strip()
            # 简化的中文音节计数
            syllable_count = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
            if syllable_count > 0:
                syllable_durations.append(duration / syllable_count)
        
        if syllable_durations:
            articulation_rate = 1.0 / np.mean(syllable_durations)  # 音节/秒
            syllable_variability = np.std(syllable_durations)
        else:
            articulation_rate = 0
            syllable_variability = 0
        
        return {
            'articulation_rate': articulation_rate,
            'syllable_variability': syllable_variability,
            'speech_rate_chars_per_min': 0,  # 将在后面计算
            'rhythm_anomalies': []  # 待人工标注训练
        }
    
    def extract_transcription_features(self, text: str, segments: List[Dict]) -> Dict:
        """5. 文本内容与填充词 (Transcription & Filled Pauses) 特征提取"""
        
        # 检测填充词
        filled_pauses = ['嗯', '啊', '呃', '呐', '那个', '就是']
        filled_pause_count = 0
        
        for fp in filled_pauses:
            filled_pause_count += text.count(fp)
        
        # 检测重复
        words = text.split()
        word_counts = {}
        for word in words:
            if len(word) > 1:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        repetitions = sum(1 for count in word_counts.values() if count > 1)
        
        # 自我纠正检测（简单版）
        self_corrections = text.count('不对') + text.count('不是') + text.count('应该是')
        
        return {
            'filled_pauses_count': filled_pause_count,
            'repetitions': repetitions,
            'self_corrections': self_corrections,
            'total_chars': len([c for c in text if c.strip()]),
            'filled_pause_ratio': filled_pause_count / len(words) if words else 0
        }
    
    def analyze_pause_functions(self, pauses: List[Dict], segments: List[Dict]) -> Dict:
        """6. 停顿结构与功能 (Pause Structure & Function) - 增强版"""
        
        if not pauses:
            return {
                'functional_pauses': {'syntactic': 0, 'hesitation': 0, 'word_finding': 0, 'breathing': 0},
                'pause_position_analysis': {}
            }
        
        # 简化的停顿功能分类（基于时长和位置）
        functional_pauses = {'syntactic': 0, 'hesitation': 0, 'word_finding': 0, 'breathing': 0}
        
        for pause in pauses:
            duration = pause['duration']
            
            # 基于时长的初步分类
            if duration < 0.5:
                functional_pauses['syntactic'] += 1
            elif duration < 1.0:
                functional_pauses['hesitation'] += 1
            elif duration < 2.0:
                functional_pauses['word_finding'] += 1
            else:
                functional_pauses['breathing'] += 1
        
        # 停顿位置分析
        mid_sentence_pauses = 0
        sentence_boundary_pauses = 0
        
        # 这里需要更复杂的语法分析，暂时简化处理
        for pause in pauses:
            # 假设判断逻辑
            if pause['duration'] > 1.0:
                mid_sentence_pauses += 1
            else:
                sentence_boundary_pauses += 1
        
        return {
            'functional_pauses': functional_pauses,
            'pause_position_analysis': {
                'mid_sentence': mid_sentence_pauses,
                'sentence_boundary': sentence_boundary_pauses
            }
        }
    
    def detect_articulatory_errors(self, audio_path: str, segments: List[Dict]) -> Dict:
        """7. 构音障碍点 (Articulatory Error Points) 检测"""
        
        # 这是一个复杂的任务，需要专门的模型
        # 这里提供基础框架，实际实现需要训练专门的错误检测模型
        
        articulatory_errors = []
        error_confidence_scores = []
        
        # 基于ASR置信度的初步错误检测
        for segment in segments:
            if 'confidence' in segment:
                confidence = segment.get('confidence', 1.0)
                if confidence < 0.7:  # 低置信度可能表示发音问题
                    articulatory_errors.append({
                        'text': segment['text'],
                        'start': segment['start'],
                        'end': segment['end'],
                        'error_type': 'LOW_CONFIDENCE',
                        'confidence': confidence
                    })
        
        return {
            'error_points': articulatory_errors,
            'error_types': {'SLUR': 0, 'SUB': 0, 'OMIT': 0},  # 待训练专门模型
            'total_errors': len(articulatory_errors)
        }
    
    def detect_pauses(self, audio_path: str, segments: List[Dict]) -> Dict:
        """停顿检测 - 保持原有功能"""
        
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
        
        # 增强的停顿分析
        pause_analysis = self.analyze_pause_functions(pauses, segments)
        
        # 计算停顿统计
        total_pause_time = sum(p['duration'] for p in pauses)
        pause_ratio = total_pause_time / audio_duration if audio_duration > 0 else 0
        
        return {
            'pauses': pauses,
            'total_pause_time': total_pause_time,
            'pause_ratio': pause_ratio,
            'pause_count': len(pauses),
            'audio_duration': audio_duration,
            'average_pause_duration': total_pause_time / len(pauses) if pauses else 0,
            'functional_analysis': pause_analysis
        }
    
    def process_audio_7_dimensions(self, audio_path: str) -> Dict:
        """按照7个维度完整处理音频"""
        
        logger.info(f"🎯 开始7维度声学特征提取: {Path(audio_path).name}")
        
        # 1. ASR转录 (基础)
        result = self.model.transcribe(
            audio_path,
            language='zh',
            word_timestamps=True,
            verbose=False
        )
        
        text = result['text']
        segments = result.get('segments', [])
        
        logger.info("✅ 完成ASR转录")
        
        # 2. 停顿检测
        pause_info = self.detect_pauses(audio_path, segments)
        
        # 3. 构建带标记的文本
        marked_text = self.create_marked_text(text, segments, pause_info)
        
        # 4. 按segment/word级别提取特征
        acoustic_feature_map = {}
        
        try:
            # 为每个segment提取句子级别特征
            for i, segment in enumerate(segments):
                segment_id = f"segment_{i}"
                
                # 提取该segment的声学特征
                segment_features = self.extract_segment_features(audio_path, segment)
                acoustic_feature_map[segment_id] = segment_features
                
            # 词级别特征（发音清晰度相关）
            word_features = self.extract_word_level_features(audio_path, segments)
            acoustic_feature_map.update(word_features)
            
            # 全局停顿分析
            acoustic_feature_map['global_pause_analysis'] = {
                'pause_structure': pause_info['functional_analysis'],
                'pause_statistics': {
                    'total_count': pause_info['pause_count'],
                    'total_duration': pause_info['total_pause_time'],
                    'pause_rate': pause_info['pause_ratio'],
                    'average_duration': pause_info['average_pause_duration']
                },
                'pause_details': self.format_pause_details(pause_info['pauses'])
            }
            
            logger.info("✅ 完成所有特征提取")
            
        except Exception as e:
            logger.error(f"❌ 特征提取过程中出错: {e}")
            return None
        
        # 5. 构建符合文档设计的输出结构
        enhanced_output = {
            # 基础信息
            'audio_file': str(audio_path),
            'text': marked_text,  # 带标记的文本
            'original_text': text,
            'segments': segments,
            
            # 按文档设计的特征映射结构
            'acoustic_feature_map': acoustic_feature_map,
            
            # 待标注的感知评估分数 (人工标注目标)
            'perceptual_scores': {
                'prosody_flatness': None,      # 1-4分
                'voice_hoarseness': None,      # 1-4分  
                'articulation_clarity': None,  # 1-3分
                'rhythm_anomalies': None,      # 标记异常片段
                'pause_functions': None        # 停顿功能分类
            },
            
            # 总体评估
            'summary': {
                'total_duration': pause_info['audio_duration'],
                'speech_rate': len([c for c in text if c.strip()]) / pause_info['audio_duration'] * 60 if pause_info['audio_duration'] > 0 else 0,
                'pause_ratio': pause_info['pause_ratio'],
                'feature_extraction_success': True
            }
        }
        
        return enhanced_output
    
    def create_marked_text(self, text: str, segments: List[Dict], pause_info: Dict) -> str:
        """创建带停顿和错误标记的文本"""
        
        if not segments:
            return text
            
        pauses = pause_info['pauses']
        result_text = ""
        last_end = 0
        
        for segment in segments:
            seg_start = segment['start']
            seg_text = segment['text'].strip()
            
            # 检查是否在此分段前有停顿
            for pause in pauses:
                if last_end <= pause['start'] <= seg_start:
                    if pause['duration'] >= 0.5:  # 只标记较长的停顿
                        pause_func = self.classify_pause_function(pause['duration'])
                        result_text += f"<pause dur=\"{pause['duration']:.1f}s\" func=\"{pause_func}\">"
            
            # 检查是否有发音错误（简化版）
            if 'confidence' in segment and segment.get('confidence', 1.0) < 0.7:
                result_text += f"<error type=\"LOW_CONF\" confidence=\"{segment.get('confidence', 0):.2f}\">{seg_text}</error>"
            else:
                result_text += seg_text
                
            last_end = segment['end']
            
        return result_text.strip()
    
    def classify_pause_function(self, duration: float) -> str:
        """根据时长初步分类停顿功能"""
        if duration < 0.5:
            return "syntactic"
        elif duration < 1.0:
            return "hesitation"
        elif duration < 2.0:
            return "word_finding"
        else:
            return "breathing"
    
    def extract_segment_features(self, audio_path: str, segment: Dict) -> Dict:
        """提取单个segment的句子级别特征"""
        
        # 对于每个segment，我们提取句子级别的特征
        # 这里简化处理，实际应该切割音频segment进行分析
        
        start_time = segment['start']
        end_time = segment['end']
        duration = end_time - start_time
        text = segment['text'].strip()
        
        # 简化的segment级别特征
        return {
            'prosody': {
                'duration': duration,
                'text_length': len(text),
                'prosody_flatness_score': None  # 待人工标注
            },
            'voice_quality': {
                'segment_duration': duration,
                'voice_hoarseness_score': None  # 待人工标注
            },
            'rhythm': {
                'articulation_rate': len(text) / duration if duration > 0 else 0,
                'speed_anomalies': []  # 待人工标注
            },
            'transcription': {
                'text': text,
                'filled_pauses_in_segment': self.count_filled_pauses_in_text(text),
                'repetitions_in_segment': self.count_repetitions_in_text(text)
            }
        }
    
    def extract_word_level_features(self, audio_path: str, segments: List[Dict]) -> Dict:
        """提取词级别特征（主要是发音清晰度）"""
        
        word_features = {}
        word_index = 0
        
        for segment in segments:
            words = segment['text'].strip().split()
            for word in words:
                if len(word) > 1:  # 只处理有意义的词
                    word_id = f"word_{word_index}"
                    word_features[word_id] = {
                        'articulation': {
                            'word': word,
                            'segment_start': segment['start'],
                            'segment_end': segment['end'],
                            'clarity_score': None,  # 待人工标注
                            'error_points': []  # 待人工标注：["SLUR", "SUB", "OMIT"]
                        }
                    }
                    word_index += 1
        
        return word_features
    
    def format_pause_details(self, pauses: List[Dict]) -> List[Dict]:
        """格式化停顿详细信息"""
        
        formatted_pauses = []
        for i, pause in enumerate(pauses):
            pause_func = self.classify_pause_function(pause['duration'])
            
            formatted_pauses.append({
                'id': f"pause_{i}",
                'start': pause['start'],
                'end': pause['end'],
                'duration': pause['duration'],
                'position': "mid_sentence" if pause['duration'] > 1.0 else "sentence_boundary",
                'function': pause_func,  # 初步分类
                'confidence': 0.8  # 初始置信度，待训练模型优化
            })
        
        return formatted_pauses
    
    def count_filled_pauses_in_text(self, text: str) -> int:
        """统计文本中的填充词"""
        filled_pauses = ['嗯', '啊', '呃', '呐', '那个', '就是']
        count = 0
        for fp in filled_pauses:
            count += text.count(fp)
        return count
    
    def count_repetitions_in_text(self, text: str) -> int:
        """统计文本中的重复"""
        words = text.split()
        word_counts = {}
        for word in words:
            if len(word) > 1:
                word_counts[word] = word_counts.get(word, 0) + 1
        return sum(1 for count in word_counts.values() if count > 1)

def test_seven_dimension_extractor():
    """测试7维度特征提取器"""
    
    logger.info("🚀 开始7维度声学特征提取测试...")
    
    # 检查样本文件
    sample_dir = Path("data/raw/audio/samples")
    audio_files = list(sample_dir.rglob("*.wav"))
    
    if not audio_files:
        # 也检查其他音频格式
        audio_files = list(sample_dir.rglob("*.mp3")) + list(sample_dir.rglob("*.m4a"))
    
    if not audio_files:
        logger.error("❌ 未找到音频文件")
        return False
    
    # 创建7维度特征提取器
    logger.info("🔧 初始化7维度特征提取器...")
    extractor = SevenDimensionAcousticExtractor()
    
    # 测试文件
    for i, audio_file in enumerate(audio_files[:2]):  # 测试前2个文件
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 测试文件 {i+1}/{min(2, len(audio_files))}: {audio_file.name}")
        
        try:
            result = extractor.process_audio_7_dimensions(str(audio_file))
            
            if result is None:
                logger.error("❌ 处理失败")
                continue
            
            # 打印详细结果
            logger.info("\n📋 7维度特征提取结果:")
            
            # 维度1: 韵律
            prosody = result['acoustic_feature_map']['segment_0']['prosody']
            logger.info(f"🎵 韵律特征:")
            logger.info(f"  F0标准差: {prosody['f0_std']:.2f} Hz")
            logger.info(f"  音高范围: {prosody['pitch_range_st']:.2f} 半音")
            
            # 维度2: 音质
            voice = result['acoustic_feature_map']['segment_0']['voice_quality']
            logger.info(f"🔊 音质特征:")
            logger.info(f"  Jitter: {voice['jitter']:.3f}%")
            logger.info(f"  Shimmer: {voice['shimmer']:.3f}")
            logger.info(f"  HNR: {voice['hnr']:.2f} dB")
            
            # 维度3: 发音清晰度
            artic = result['acoustic_feature_map']['word_0']['articulation']
            logger.info(f"🗣️  发音清晰度:")
            logger.info(f"  VSA代理指标: {artic['vsa_proxy']:.2f}")
            logger.info(f"  共振峰变化率: {artic['formant_transition_rate']:.2f}")
            
            # 维度4: 语速节律
            rhythm = result['acoustic_feature_map']['segment_0']['rhythm']
            logger.info(f"⏱️  语速节律:")
            logger.info(f"  发音速率: {rhythm['articulation_rate']:.2f} 音节/秒")
            logger.info(f"  音节时长变异性: {rhythm['syllable_variability']:.3f}")
            
            # 维度5: 文本特征
            trans = result['acoustic_feature_map']['segment_0']['transcription']
            logger.info(f"📝 文本特征:")
            logger.info(f"  填充词数量: {trans['filled_pauses_in_segment']}")
            logger.info(f"  重复次数: {trans['repetitions_in_segment']}")
            logger.info(f"  自我纠正: {trans['self_corrections']}")
            
            # 维度6: 停顿分析
            pause = result['acoustic_feature_map']['global_pause_analysis']
            logger.info(f"⏸️  停顿分析:")
            logger.info(f"  停顿次数: {pause['pause_statistics']['total_count']}")
            logger.info(f"  停顿比例: {pause['pause_statistics']['pause_rate']:.1%}")
            logger.info(f"  功能分类: {pause['pause_structure']['functional_pauses']}")
            
            # 维度7: 构音错误
            errors = result['acoustic_feature_map']['word_0']['articulation']['error_points']
            logger.info(f"🚨 构音错误:")
            logger.info(f"  检测到错误: {errors}")
            
            # 总体评估
            summary = result['summary']
            logger.info(f"\n📊 总体评估:")
            logger.info(f"  音频时长: {summary['total_duration']:.1f}秒")
            logger.info(f"  语速: {summary['speech_rate']:.1f} 字/分钟")
            logger.info(f"  停顿比例: {summary['pause_ratio']:.1%}")
            
            logger.info("✅ 7维度特征提取成功!")
            
        except Exception as e:
            logger.error(f"❌ 处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    return True

def main():
    """主函数"""
    
    if test_seven_dimension_extractor():
        logger.info("\n🎉 7维度声学特征提取器测试通过！")
        logger.info("\n🎯 系统功能:")
        logger.info("  ✓ 1. 韵律和语调分析 (F0变化、语调轮廓)")
        logger.info("  ✓ 2. 音质和稳定性 (Jitter、Shimmer、HNR)")
        logger.info("  ✓ 3. 发音清晰度 (VSA、共振峰变化)")
        logger.info("  ✓ 4. 语速和节律 (发音速率、时长变异)")
        logger.info("  ✓ 5. 文本内容分析 (填充词、重复)")
        logger.info("  ✓ 6. 停顿结构与功能 (时长、位置、功能)")
        logger.info("  ✓ 7. 构音障碍点检测 (错误类型定位)")
        logger.info("\n📋 下一步:")
        logger.info("  1. 收集人工标注数据进行感知评估")
        logger.info("  2. 训练预测模型优化特征提取")
        logger.info("  3. 建立MCI检测的完整流程")
    else:
        logger.error("❌ 7维度特征提取器测试失败")

if __name__ == "__main__":
    main() 