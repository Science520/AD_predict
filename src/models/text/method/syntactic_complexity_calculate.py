#!/usr/bin/env python3
"""
句法复杂度计算模块 - 基于多种权威方法的综合计算
"""

import numpy as np
import jieba
import jieba.posseg as pseg
from typing import Dict, List
import re
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class SyntacticComplexityCalculator:
    """句法复杂度计算器"""
    
    def __init__(self):
        # 初始化jieba
        jieba.setLogLevel(logging.INFO)
    
    def _split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        sentences = re.split(r'[。！？；]', text)
        return [sent.strip() for sent in sentences if sent.strip()]
    
    def calculate_dependency_complexity(self, text: str) -> Dict[str, float]:
        """基于依存句法分析的复杂度计算"""
        try:
            import spacy
            
            # 使用spaCy进行依存句法分析
            nlp = spacy.load("zh_core_web_sm")
            doc = nlp(text)
            
            complexity_metrics = {}
            
            for sent in doc.sents:
                # 1. 依存距离计算
                dependency_distances = []
                for token in sent:
                    if token.head != token:  # 排除root
                        distance = abs(token.i - token.head.i)
                        dependency_distances.append(distance)
                
                if dependency_distances:
                    complexity_metrics['mean_dependency_distance'] = np.mean(dependency_distances)
                    complexity_metrics['max_dependency_distance'] = max(dependency_distances)
                
                # 2. 句法树深度
                tree_depths = []
                for token in sent:
                    depth = self._calculate_tree_depth(token)
                    tree_depths.append(depth)
                
                complexity_metrics['mean_tree_depth'] = np.mean(tree_depths)
                complexity_metrics['max_tree_depth'] = max(tree_depths)
                
                # 3. 依存关系类型多样性
                dep_types = [token.dep_ for token in sent if token.dep_ != 'ROOT']
                complexity_metrics['dependency_diversity'] = len(set(dep_types)) / len(dep_types) if dep_types else 0
            
            return complexity_metrics
        
        except ImportError:
            logger.warning("spaCy未安装，跳过依存句法分析")
            return {
                'mean_dependency_distance': 1.0,
                'max_dependency_distance': 1.0,
                'mean_tree_depth': 1.0,
                'max_tree_depth': 1.0,
                'dependency_diversity': 0.5
            }
    
    def _calculate_tree_depth(self, token) -> int:
        """计算token在句法树中的深度"""
        depth = 0
        current = token
        while current.head != current:
            depth += 1
            current = current.head
        return depth

    def calculate_subordinate_complexity(self, text: str) -> Dict[str, float]:
        """基于从句结构的复杂度计算"""
        
        # 中文从句标志词
        subordinate_markers = {
            'temporal': ['当', '在', '之前', '之后', '之时', '同时', '随着'],
            'causal': ['因为', '由于', '既然', '因'],
            'conditional': ['如果', '假如', '要是', '倘若', '若是'],
            'concessive': ['虽然', '尽管', '纵然', '即使'],
            'purpose': ['为了', '以便', '以免'],
            'relative': ['的', '所', '之'],
            'complement': ['说', '认为', '觉得', '知道', '听说']
        }
        
        sentences = self._split_sentences(text)
        complexity_metrics = {}
        
        total_subordinates = 0
        subordinate_types = set()
        max_embedding_depth = 0
        
        for sentence in sentences:
            # 计算从句数量和类型
            for clause_type, markers in subordinate_markers.items():
                for marker in markers:
                    count = sentence.count(marker)
                    if count > 0:
                        total_subordinates += count
                        subordinate_types.add(clause_type)
            
            # 计算嵌套深度
            embedding_depth = self._calculate_embedding_depth(sentence)
            max_embedding_depth = max(max_embedding_depth, embedding_depth)
        
        complexity_metrics['subordinate_clause_ratio'] = total_subordinates / len(sentences) if sentences else 0
        complexity_metrics['subordinate_type_diversity'] = len(subordinate_types)
        complexity_metrics['max_embedding_depth'] = max_embedding_depth
        
        return complexity_metrics

    def _calculate_embedding_depth(self, sentence: str) -> int:
        """计算嵌套深度"""
        # 简化的嵌套计算，基于标点符号
        depth = 0
        max_depth = 0
        
        for char in sentence:
            if char in ['（', '(']:
                depth += 1
                max_depth = max(max_depth, depth)
            elif char in ['）', ')']:
                depth = max(0, depth - 1)
        
        return max_depth

    def calculate_tunit_complexity(self, text: str) -> Dict[str, float]:
        """基于T-unit的复杂度计算"""
        
        # T-unit分割（基于主要标点符号）
        tunits = self._split_tunits(text)
        
        complexity_metrics = {}
        
        # 1. 平均T-unit长度
        tunit_lengths = [len(jieba.lcut(tunit)) for tunit in tunits]
        complexity_metrics['mean_tunit_length'] = np.mean(tunit_lengths) if tunit_lengths else 0
        
        # 2. 从句密度
        total_clauses = 0
        for tunit in tunits:
            # 基于连接词计算从句数
            clause_markers = ['，', '；', '因为', '虽然', '如果', '当', '的时候']
            clause_count = 1  # 主句
            for marker in clause_markers:
                clause_count += tunit.count(marker)
            total_clauses += clause_count
        
        complexity_metrics['clauses_per_tunit'] = total_clauses / len(tunits) if tunits else 0
        
        # 3. 句法复杂度指数（结合多个维度）
        complexity_metrics['syntactic_complexity_index'] = (
            complexity_metrics['mean_tunit_length'] * 0.4 +
            complexity_metrics['clauses_per_tunit'] * 0.6
        )
        
        return complexity_metrics

    def _split_tunits(self, text: str) -> List[str]:
        """分割T-units（终结单位）"""
        # 基于句号、感叹号、问号分割
        tunits = re.split(r'[。！？]', text)
        return [tunit.strip() for tunit in tunits if tunit.strip()]

    def calculate_information_complexity(self, text: str) -> Dict[str, float]:
        """基于信息论的复杂度计算"""
        import gzip
        
        complexity_metrics = {}
        
        # 1. Kolmogorov复杂度（压缩比率）
        text_bytes = text.encode('utf-8')
        compressed = gzip.compress(text_bytes)
        complexity_metrics['kolmogorov_complexity'] = len(compressed) / len(text_bytes)
        
        # 2. 词汇信息熵
        words = list(jieba.cut(text))
        word_freq = Counter(words)
        total_words = len(words)
        
        entropy = 0
        for count in word_freq.values():
            prob = count / total_words
            entropy -= prob * np.log2(prob)
        
        complexity_metrics['lexical_entropy'] = entropy
        
        # 3. 句法结构信息熵
        pos_tags = [word.flag for word in pseg.cut(text)]
        pos_freq = Counter(pos_tags)
        total_pos = len(pos_tags)
        
        pos_entropy = 0
        for count in pos_freq.values():
            prob = count / total_pos
            pos_entropy -= prob * np.log2(prob)
        
        complexity_metrics['syntactic_entropy'] = pos_entropy
        
        return complexity_metrics

    def calculate_comprehensive_syntactic_complexity(self, text: str) -> float:
        """综合多种权威方法的句法复杂度计算"""
        
        # 获取各项指标
        dep_metrics = self.calculate_dependency_complexity(text)
        sub_metrics = self.calculate_subordinate_complexity(text)
        tunit_metrics = self.calculate_tunit_complexity(text)
        info_metrics = self.calculate_information_complexity(text)
        
        # 权重分配（基于文献重要性）
        weights = {
            'dependency': 0.3,      # 依存距离权重
            'subordinate': 0.25,    # 从句结构权重
            'tunit': 0.25,         # T-unit权重
            'information': 0.2      # 信息论权重
        }
        
        # 标准化各指标到0-10范围
        normalized_scores = {}
        
        # 依存复杂度 (较高的依存距离表示更复杂)
        normalized_scores['dependency'] = min(
            dep_metrics.get('mean_dependency_distance', 1) * 2, 10
        )
        
        # 从句复杂度
        normalized_scores['subordinate'] = min(
            (sub_metrics.get('subordinate_clause_ratio', 0) * 5 + 
             sub_metrics.get('subordinate_type_diversity', 0) * 0.5 +
             sub_metrics.get('max_embedding_depth', 0) * 2), 10
        )
        
        # T-unit复杂度
        normalized_scores['tunit'] = min(
            tunit_metrics.get('syntactic_complexity_index', 0), 10
        )
        
        # 信息复杂度
        normalized_scores['information'] = min(
            (info_metrics.get('lexical_entropy', 0) * 0.5 +
             info_metrics.get('syntactic_entropy', 0) * 0.5), 10
        )
        
        # 加权平均
        final_score = sum(
            normalized_scores[key] * weights[key] 
            for key in weights.keys()
        )
        
        return min(final_score, 10.0)