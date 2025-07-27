import torch
import numpy as np
import nltk
import spacy
import textstat
import re
from typing import Dict, List, Optional
import logging
from collections import Counter
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

# 确保nltk数据已下载
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class TextProcessor:
    """文本处理器
    
    负责文本预处理、特征提取和语言学分析
    """
    
    def __init__(self, text_config: Dict):
        """
        Args:
            text_config: 文本处理配置
        """
        self.config = text_config
        self.max_length = text_config['max_length']
        self.min_word_count = text_config['min_word_count']
        self.remove_stopwords = text_config['remove_stopwords']
        self.lemmatize = text_config['lemmatize']
        self.language = text_config['language']
        
        # 初始化tokenizer和模型
        try:
            model_name = text_config.get('model_name', 'bert-base-uncased')
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
        except Exception as e:
            logger.warning(f"无法加载预训练模型: {e}")
            self.tokenizer = None
            self.model = None
        
        # 初始化spacy模型
        try:
            if self.language == 'en':
                self.nlp = spacy.load('en_core_web_sm')
            elif self.language == 'zh':
                self.nlp = spacy.load('zh_core_web_sm')
            else:
                self.nlp = None
                logger.warning(f"不支持的语言: {self.language}")
        except Exception as e:
            logger.warning(f"无法加载spacy模型: {e}")
            self.nlp = None
        
        logger.info(f"初始化文本处理器: language={self.language}, max_length={self.max_length}")
    
    def process(self, text: str) -> torch.Tensor:
        """处理文本并提取特征
        
        Args:
            text: 输入文本
            
        Returns:
            文本特征张量 [D]
        """
        try:
            # 预处理文本
            cleaned_text = self._preprocess_text(text)
            
            if self.model and self.tokenizer:
                # 使用预训练模型提取特征
                features = self._extract_bert_features(cleaned_text)
            else:
                # 使用传统特征
                features = self._extract_traditional_features(cleaned_text)
            
            return features
            
        except Exception as e:
            logger.error(f"文本处理失败: {e}")
            # 返回零特征
            feature_dim = self.config.get('feature_dim', 768)
            return torch.zeros(feature_dim)
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        try:
            # 基本清理
            text = re.sub(r'<[^>]+>', ' ', text)  # 移除HTML标签
            text = re.sub(r'[^\w\s\.\?\!,;:]', ' ', text)  # 保留基本标点
            text = re.sub(r'\s+', ' ', text)  # 多个空格合并为一个
            text = text.strip()
            
            # 处理停顿标记
            text = self._process_pause_marks(text)
            
            return text
            
        except Exception as e:
            logger.error(f"文本预处理失败: {e}")
            return text
    
    def _process_pause_marks(self, text: str) -> str:
        """处理停顿标记"""
        # 将各种停顿标记统一为<pause>
        pause_patterns = [
            r'\[pause\]', r'\(pause\)', r'<pause>', r'\.\.\.+', r'--+', r'__+'
        ]
        
        for pattern in pause_patterns:
            text = re.sub(pattern, '<pause>', text, flags=re.IGNORECASE)
        
        return text
    
    def _extract_bert_features(self, text: str) -> torch.Tensor:
        """使用BERT提取特征"""
        try:
            # 分词和编码
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=self.max_length,
                padding='max_length',
                truncation=True
            )
            
            # 提取特征
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用[CLS]标记的特征作为句子表示
                features = outputs.last_hidden_state[:, 0, :]  # [1, hidden_dim]
                features = features.squeeze(0)  # [hidden_dim]
            
            return features
            
        except Exception as e:
            logger.error(f"BERT特征提取失败: {e}")
            feature_dim = self.config.get('feature_dim', 768)
            return torch.zeros(feature_dim)
    
    def _extract_traditional_features(self, text: str) -> torch.Tensor:
        """提取传统文本特征"""
        try:
            # 基本统计特征
            word_count = len(text.split())
            char_count = len(text)
            sentence_count = len(nltk.sent_tokenize(text))
            
            # 词汇丰富度特征
            lexical_features = self.calculate_lexical_richness(text)
            
            # 句法复杂度特征
            syntactic_features = self.calculate_syntactic_complexity(text)
            
            # 停顿特征
            pause_features = self.extract_pause_features(text)
            
            # 合并所有特征
            all_features = [
                word_count / 100,  # 归一化
                char_count / 1000,
                sentence_count / 10,
                lexical_features['ttr'],
                lexical_features['mtld'],
                syntactic_features['avg_sentence_length'],
                syntactic_features['avg_tree_depth'],
                pause_features['pause_ratio'],
                pause_features['pause_count'] / 10,
            ]
            
            # 填充到指定维度
            feature_dim = self.config.get('feature_dim', 768)
            if len(all_features) < feature_dim:
                all_features.extend([0.0] * (feature_dim - len(all_features)))
            else:
                all_features = all_features[:feature_dim]
            
            return torch.tensor(all_features, dtype=torch.float32)
            
        except Exception as e:
            logger.error(f"传统特征提取失败: {e}")
            feature_dim = self.config.get('feature_dim', 768)
            return torch.zeros(feature_dim)
    
    def calculate_lexical_richness(self, text: str) -> Dict[str, float]:
        """计算词汇丰富度"""
        try:
            words = text.lower().split()
            
            if len(words) == 0:
                return {'ttr': 0.0, 'mtld': 0.0, 'vocd': 0.0}
            
            # Type-Token Ratio (TTR)
            unique_words = set(words)
            ttr = len(unique_words) / len(words)
            
            # Measure of Textual Lexical Diversity (MTLD) - 简化版本
            mtld = self._calculate_mtld_simple(words)
            
            # VocD - 使用简化的词汇多样性指标
            vocd = self._calculate_vocd_simple(words)
            
            return {
                'ttr': float(ttr),
                'mtld': float(mtld),
                'vocd': float(vocd)
            }
            
        except Exception as e:
            logger.error(f"词汇丰富度计算失败: {e}")
            return {'ttr': 0.0, 'mtld': 0.0, 'vocd': 0.0}
    
    def _calculate_mtld_simple(self, words: List[str]) -> float:
        """简化的MTLD计算"""
        try:
            if len(words) < 50:
                return len(set(words)) / len(words) if words else 0
            
            # 滑动窗口计算TTR
            window_size = 50
            ttrs = []
            
            for i in range(len(words) - window_size + 1):
                window = words[i:i + window_size]
                window_ttr = len(set(window)) / len(window)
                ttrs.append(window_ttr)
            
            return sum(ttrs) / len(ttrs) if ttrs else 0
            
        except Exception:
            return 0.0
    
    def _calculate_vocd_simple(self, words: List[str]) -> float:
        """简化的VocD计算"""
        try:
            word_freq = Counter(words)
            
            # 计算低频词比例 (频次为1的词)
            hapax_count = sum(1 for freq in word_freq.values() if freq == 1)
            vocd = hapax_count / len(word_freq) if word_freq else 0
            
            return vocd
            
        except Exception:
            return 0.0
    
    def calculate_syntactic_complexity(self, text: str) -> Dict[str, float]:
        """计算句法复杂度"""
        try:
            if not self.nlp:
                return self._calculate_syntactic_simple(text)
            
            doc = self.nlp(text)
            
            # 句子长度统计
            sentence_lengths = []
            tree_depths = []
            
            for sent in doc.sents:
                sentence_lengths.append(len([token for token in sent if not token.is_punct]))
                
                # 计算句法树深度
                depth = self._calculate_tree_depth(sent.root)
                tree_depths.append(depth)
            
            avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
            avg_tree_depth = sum(tree_depths) / len(tree_depths) if tree_depths else 0
            
            # 从句比例
            subordinate_clauses = len([token for token in doc if token.dep_ in ['ccomp', 'xcomp', 'advcl']])
            subordination_ratio = subordinate_clauses / len(list(doc.sents)) if list(doc.sents) else 0
            
            return {
                'avg_sentence_length': float(avg_sentence_length),
                'avg_tree_depth': float(avg_tree_depth),
                'subordination_ratio': float(subordination_ratio)
            }
            
        except Exception as e:
            logger.error(f"句法复杂度计算失败: {e}")
            return self._calculate_syntactic_simple(text)
    
    def _calculate_syntactic_simple(self, text: str) -> Dict[str, float]:
        """简化的句法复杂度计算"""
        try:
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            
            avg_sentence_length = len(words) / len(sentences) if sentences else 0
            
            # 使用标点符号估算句法复杂度
            complex_punct_count = text.count(',') + text.count(';') + text.count(':')
            avg_tree_depth = 1 + (complex_punct_count / len(sentences)) if sentences else 1
            
            return {
                'avg_sentence_length': float(avg_sentence_length),
                'avg_tree_depth': float(avg_tree_depth),
                'subordination_ratio': 0.0
            }
            
        except Exception:
            return {
                'avg_sentence_length': 0.0,
                'avg_tree_depth': 1.0,
                'subordination_ratio': 0.0
            }
    
    def _calculate_tree_depth(self, token) -> int:
        """递归计算句法树深度"""
        try:
            if not list(token.children):
                return 1
            
            max_child_depth = max([self._calculate_tree_depth(child) for child in token.children])
            return 1 + max_child_depth
            
        except Exception:
            return 1
    
    def extract_pause_features(self, text: str) -> Dict[str, float]:
        """提取停顿相关特征"""
        try:
            # 统计停顿标记
            pause_count = text.count('<pause>')
            
            # 移除停顿标记后的文本
            clean_text = text.replace('<pause>', ' ')
            words = clean_text.split()
            
            # 计算停顿比例
            total_elements = len(words) + pause_count
            pause_ratio = pause_count / total_elements if total_elements > 0 else 0
            
            # 平均停顿间隔
            if pause_count > 0:
                avg_pause_interval = len(words) / pause_count
            else:
                avg_pause_interval = len(words)
            
            return {
                'pause_count': float(pause_count),
                'pause_ratio': float(pause_ratio),
                'avg_pause_interval': float(avg_pause_interval)
            }
            
        except Exception as e:
            logger.error(f"停顿特征提取失败: {e}")
            return {
                'pause_count': 0.0,
                'pause_ratio': 0.0,
                'avg_pause_interval': 0.0
            }
    
    def calculate_readability_scores(self, text: str) -> Dict[str, float]:
        """计算可读性分数"""
        try:
            # 移除停顿标记
            clean_text = text.replace('<pause>', ' ')
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            if len(clean_text) < 10:
                return {
                    'flesch_reading_ease': 0.0,
                    'flesch_kincaid_grade': 0.0,
                    'automated_readability_index': 0.0
                }
            
            # 计算各种可读性指标
            flesch_reading_ease = textstat.flesch_reading_ease(clean_text)
            flesch_kincaid_grade = textstat.flesch_kincaid_grade(clean_text)
            automated_readability_index = textstat.automated_readability_index(clean_text)
            
            return {
                'flesch_reading_ease': float(flesch_reading_ease),
                'flesch_kincaid_grade': float(flesch_kincaid_grade),
                'automated_readability_index': float(automated_readability_index)
            }
            
        except Exception as e:
            logger.error(f"可读性分数计算失败: {e}")
            return {
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0,
                'automated_readability_index': 0.0
            }
    
    def calculate_semantic_coherence(self, text: str) -> float:
        """计算语义连贯性 (简化版本)"""
        try:
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < 2:
                return 1.0
            
            if self.model and self.tokenizer:
                # 使用BERT计算句子间相似度
                sentence_embeddings = []
                
                for sent in sentences:
                    inputs = self.tokenizer(
                        sent,
                        return_tensors='pt',
                        max_length=128,
                        padding=True,
                        truncation=True
                    )
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                        sentence_embeddings.append(embedding)
                
                # 计算相邻句子的余弦相似度
                similarities = []
                for i in range(len(sentence_embeddings) - 1):
                    sim = torch.cosine_similarity(
                        sentence_embeddings[i], 
                        sentence_embeddings[i + 1], 
                        dim=0
                    )
                    similarities.append(sim.item())
                
                return float(sum(similarities) / len(similarities)) if similarities else 0.0
            else:
                # 使用词汇重叠作为简单的连贯性度量
                overlaps = []
                for i in range(len(sentences) - 1):
                    words1 = set(sentences[i].lower().split())
                    words2 = set(sentences[i + 1].lower().split())
                    
                    intersection = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    
                    overlap = intersection / union if union > 0 else 0
                    overlaps.append(overlap)
                
                return float(sum(overlaps) / len(overlaps)) if overlaps else 0.0
                
        except Exception as e:
            logger.error(f"语义连贯性计算失败: {e}")
            return 0.0 