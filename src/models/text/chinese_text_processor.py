#!/usr/bin/env python3
"""
中文文本处理模型 - 支持医学概念提取和老年人语言特点分析
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import jieba
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import re
import json
from pathlib import Path
from dataclasses import dataclass
from .method.syntactic_complexity_calculate import SyntacticComplexityCalculator

logger = logging.getLogger(__name__)

@dataclass
class TextFeatures:
    """文本特征输出"""
    embeddings: torch.Tensor  # 文本嵌入
    linguistic_features: Dict  # 语言学特征
    medical_concepts: List[str]  # 医学概念
    elderly_patterns: Dict  # 老年人语言模式

class ChineseTextProcessor(nn.Module):
    """中文文本处理器 - 针对医学领域和老年人语言优化"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 加载预训练模型
        model_name = config.get('model_name', 'hfl/chinese-roberta-wwm-ext')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.max_length = config.get('max_length', 512)
        
        # 医学概念配置
        medical_config = config.get('medical_concepts', {})
        self.enable_medical_ner = medical_config.get('enable_medical_ner', True)
        
        # 老年人语言特点配置
        elderly_config = config.get('elderly_language', {})
        self.dialect_support = elderly_config.get('dialect_support', True)
        self.repetition_detection = elderly_config.get('repetition_detection', True)
        self.coherence_analysis = elderly_config.get('coherence_analysis', True)
        
        # 加载医学词汇表
        self.medical_terms = self._load_medical_terms(medical_config.get('medical_vocab_path'))
        
        # 初始化jieba分词
        jieba.setLogLevel(logging.INFO)
        
        # 初始化句法复杂度计算器
        self.syntax_calculator = SyntacticComplexityCalculator()
        
    def _load_medical_terms(self, vocab_path: Optional[str]) -> List[str]:
        """加载医学术语词汇表"""
        
        if not vocab_path or not Path(vocab_path).exists():
            # 默认医学术语（认知相关）
            return [
                '记忆力', '认知', '痴呆', '阿尔茨海默', '帕金森',
                '脑梗', '中风', '头晕', '健忘', '糊涂',
                '说话', '语言', '表达', '理解', '思维',
                '注意力', '集中', '分心', '困惑', '清醒'
            ]
        
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.warning(f"无法加载医学词汇表 {vocab_path}: {e}")
            return []
    
    def extract_linguistic_features(self, text: str) -> Dict:
        """提取语言学特征"""
        
        # 分词
        words = list(jieba.cut(text))
        words = [w for w in words if w.strip()]
        
        # 基本统计
        char_count = len([c for c in text if c.strip()])
        word_count = len(words)
        sentence_count = len(re.findall(r'[。！？；]', text))
        
        # Type-Token Ratio (词汇丰富度)
        unique_words = set(words)
        ttr = len(unique_words) / word_count if word_count > 0 else 0
        
        # 平均句长
        avg_sentence_length = char_count / sentence_count if sentence_count > 0 else 0
        
        # 停顿统计
        pause_markers = re.findall(r'<pause:(\d+\.?\d*)s>', text)
        pause_count = len(pause_markers)
        total_pause_time = sum(float(p) for p in pause_markers)
        
        # 停顿比例
        pause_ratio = pause_count / (word_count + pause_count) if (word_count + pause_count) > 0 else 0
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'ttr': ttr,  # 词汇丰富度
            'avg_sentence_length': avg_sentence_length,
            'pause_count': pause_count,
            'total_pause_time': total_pause_time,
            'pause_ratio': pause_ratio,
            'unique_word_count': len(unique_words)
        }
    
    def extract_medical_concepts(self, text: str) -> List[str]:
        """提取医学概念"""
        
        if not self.enable_medical_ner:
            return []
        
        concepts = []
        
        # 简单的关键词匹配
        for term in self.medical_terms:
            if term in text:
                concepts.append(term)
        
        # 症状描述模式
        symptom_patterns = [
            r'记不住',
            r'想不起',
            r'糊涂',
            r'健忘',
            r'说不出',
            r'找不到词',
            r'头疼',
            r'头晕',
            r'犯糊涂'
        ]
        
        for pattern in symptom_patterns:
            if re.search(pattern, text):
                concepts.append(pattern)
        
        return list(set(concepts))  # 去重
    
    def analyze_elderly_patterns(self, text: str) -> Dict:
        """分析老年人语言模式"""
        
        patterns = {}
        
        if self.repetition_detection:
            # 重复检测
            patterns.update(self._detect_repetitions(text))
        
        if self.coherence_analysis:
            # 连贯性分析
            patterns.update(self._analyze_coherence(text))
        
        if self.dialect_support:
            # 方言特征检测
            patterns.update(self._detect_dialect_features(text))
        
        return patterns
    
    def _detect_repetitions(self, text: str) -> Dict:
        """检测重复模式"""
        
        words = list(jieba.cut(text))
        words = [w for w in words if w.strip() and len(w) > 1]
        
        # 词汇重复
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        repeated_words = {word: count for word, count in word_counts.items() if count > 1}
        
        # 短语重复检测
        repeated_phrases = []
        for i in range(len(words) - 1):
            phrase = words[i] + words[i + 1]
            if text.count(phrase) > 1:
                repeated_phrases.append(phrase)
        
        return {
            'repeated_words': repeated_words,
            'repeated_phrases': list(set(repeated_phrases)),
            'repetition_ratio': len(repeated_words) / len(set(words)) if words else 0
        }
    
    def _analyze_coherence(self, text: str) -> Dict:
        """分析连贯性"""
        
        sentences = re.split(r'[。！？；]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 句子长度变异
        if sentences:
            lengths = [len(s) for s in sentences]
            length_variance = np.var(lengths) if len(lengths) > 1 else 0
            avg_length = np.mean(lengths)
        else:
            length_variance = 0
            avg_length = 0
        
        # 话题连贯性 (简单指标)
        # 检查是否有明显的话题跳转词汇
        topic_shift_markers = ['突然', '另外', '对了', '还有', '忘了说']
        topic_shifts = sum(1 for marker in topic_shift_markers if marker in text)
        
        return {
            'sentence_count': len(sentences),
            'length_variance': length_variance,
            'avg_sentence_length': avg_length,
            'topic_shifts': topic_shifts,
            'coherence_score': 1.0 / (1.0 + topic_shifts)  # 简单的连贯性分数
        }
    
    def _detect_dialect_features(self, text: str) -> Dict:
        """检测方言特征"""
        # 这种还是放到asr之后二次重排序？
        
        # 常见方言词汇或表达
        dialect_markers = {
            '北方方言': ['俺', '咱', '嘛', '呗'],
            '南方方言': ['侬', '阿拉', '蛮', '格'],
            '四川方言': ['撒子', '哦豁', '巴适'],
            '东北方言': ['整', '嘎哈', '老铁'],
        }
        
        detected_dialects = {}
        for dialect, markers in dialect_markers.items():
            count = sum(1 for marker in markers if marker in text)
            if count > 0:
                detected_dialects[dialect] = count
        
        return {
            'detected_dialects': detected_dialects,
            'has_dialect_features': len(detected_dialects) > 0
        }
    
    
    def forward(self, text: str) -> TextFeatures:
        """前向传播 - 完整的文本处理流程"""
        
        try:
            # 1. 文本编码
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            # 2. 获取文本嵌入
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # 平均池化
            
            # 3. 提取语言学特征
            linguistic_features = self.extract_linguistic_features(text)
            
            # 4. 计算句法复杂度, 综合常见四个方法见文件syntactic_complexity_calculate.py
            linguistic_features['syntactic_complexity'] = self.syntax_calculator.calculate_comprehensive_syntactic_complexity(text)
            
            # 5. 提取医学概念
            medical_concepts = self.extract_medical_concepts(text)
            
            # 6. 分析老年人语言模式
            elderly_patterns = self.analyze_elderly_patterns(text)
            
            return TextFeatures(
                embeddings=embeddings,
                linguistic_features=linguistic_features,
                medical_concepts=medical_concepts,
                elderly_patterns=elderly_patterns
            )
            
        except Exception as e:
            logger.error(f"文本处理失败: {e}")
            # 返回空结果
            return TextFeatures(
                embeddings=torch.zeros(1, self.config.get('feature_dim', 768)),
                linguistic_features={},
                medical_concepts=[],
                elderly_patterns={}
            )
    
    def batch_process(self, texts: List[str]) -> List[TextFeatures]:
        """批量处理文本"""
        
        results = []
        for text in texts:
            result = self.forward(text)
            results.append(result)
            
        return results

def create_chinese_text_processor(config: Dict) -> ChineseTextProcessor:
    """创建中文文本处理器实例"""
    
    processor = ChineseTextProcessor(config)
    
    logger.info(f"创建中文文本处理器: {config.get('model_name')}")
    logger.info(f"医学概念提取: {config.get('medical_concepts', {}).get('enable_medical_ner', True)}")
    logger.info(f"老年人语言分析: {config.get('elderly_language', {})}")
    
    return processor 