import torch
import numpy as np
import mne
from scipy import signal
from scipy.signal import butter, filtfilt
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class EEGProcessor:
    """EEG信号处理器
    
    负责EEG信号的加载、预处理和特征提取
    """
    
    def __init__(self, eeg_config: Dict):
        """
        Args:
            eeg_config: EEG处理配置
        """
        self.config = eeg_config
        self.sample_rate = eeg_config['sample_rate']
        self.lowpass_freq = eeg_config['lowpass_freq']
        self.highpass_freq = eeg_config['highpass_freq']
        self.notch_freq = eeg_config['notch_freq']
        self.window_length = eeg_config['window_length']
        self.overlap = eeg_config['overlap']
        self.channels = eeg_config['channels']
        
        # 频段定义
        self.frequency_bands = {
            'delta': [0.5, 4],
            'theta': [4, 8],
            'alpha': [8, 13], 
            'beta': [13, 30],
            'gamma': [30, 50]  # 限制在低通滤波频率内
        }
        
        logger.info(f"初始化EEG处理器: sr={self.sample_rate}Hz, channels={len(self.channels)}")
    
    def process(self, eeg_path: str) -> torch.Tensor:
        """处理EEG文件
        
        Args:
            eeg_path: EEG文件路径
            
        Returns:
            EEG特征张量 [T, C, D] (时间, 通道, 特征维度)
        """
        try:
            # 加载EEG数据
            raw_data = self._load_eeg_file(eeg_path)
            
            if raw_data is None:
                logger.warning(f"无法加载EEG文件: {eeg_path}")
                return self._get_dummy_features()
            
            # 预处理
            processed_data = self._preprocess(raw_data)
            
            # 特征提取
            features = self._extract_features(processed_data)
            
            return torch.tensor(features, dtype=torch.float32)
            
        except Exception as e:
            logger.error(f"处理EEG文件 {eeg_path} 时出错: {e}")
            return self._get_dummy_features()
    
    def _load_eeg_file(self, eeg_path: str) -> Optional[np.ndarray]:
        """加载EEG文件"""
        try:
            file_path = Path(eeg_path)
            
            if not file_path.exists():
                logger.warning(f"EEG文件不存在: {eeg_path}")
                return None
            
            # 支持多种EEG文件格式
            if file_path.suffix.lower() == '.edf':
                raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)
            elif file_path.suffix.lower() == '.fif':
                raw = mne.io.read_raw_fif(eeg_path, preload=True, verbose=False)
            elif file_path.suffix.lower() in ['.csv', '.txt']:
                # 简单的CSV格式
                data = np.loadtxt(eeg_path, delimiter=',')
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                return data
            else:
                logger.error(f"不支持的EEG文件格式: {file_path.suffix}")
                return None
            
            # 选择感兴趣的通道
            available_channels = raw.ch_names
            selected_channels = [ch for ch in self.channels if ch in available_channels]
            
            if not selected_channels:
                logger.warning("没有找到匹配的EEG通道，使用前几个通道")
                selected_channels = available_channels[:len(self.channels)]
            
            # 选择通道并获取数据
            raw.pick_channels(selected_channels)
            data = raw.get_data()
            
            # 重采样
            if raw.info['sfreq'] != self.sample_rate:
                data = self._resample(data, raw.info['sfreq'], self.sample_rate)
            
            return data
            
        except Exception as e:
            logger.error(f"加载EEG文件失败: {e}")
            return None
    
    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """EEG信号预处理
        
        Args:
            data: 原始EEG数据 [C, T]
            
        Returns:
            预处理后的数据 [C, T]
        """
        try:
            # 带通滤波
            data = self._apply_bandpass_filter(data)
            
            # 工频滤波
            data = self._apply_notch_filter(data)
            
            # 伪迹去除 (简化版本)
            data = self._remove_artifacts(data)
            
            # 标准化
            data = self._standardize(data)
            
            return data
            
        except Exception as e:
            logger.error(f"EEG预处理失败: {e}")
            return data
    
    def _apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """应用带通滤波器"""
        try:
            nyquist = self.sample_rate / 2
            low = self.highpass_freq / nyquist
            high = self.lowpass_freq / nyquist
            
            b, a = butter(4, [low, high], btype='band')
            
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[0]):
                filtered_data[i] = filtfilt(b, a, data[i])
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"带通滤波失败: {e}")
            return data
    
    def _apply_notch_filter(self, data: np.ndarray) -> np.ndarray:
        """应用工频滤波器"""
        try:
            # 50Hz工频滤波
            nyquist = self.sample_rate / 2
            freq = self.notch_freq / nyquist
            quality = 30.0
            
            b, a = signal.iirnotch(freq, quality)
            
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[0]):
                filtered_data[i] = filtfilt(b, a, data[i])
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"工频滤波失败: {e}")
            return data
    
    def _remove_artifacts(self, data: np.ndarray) -> np.ndarray:
        """去除伪迹 (简化版本)"""
        try:
            # 幅值阈值去除
            threshold = 5 * np.std(data, axis=1, keepdims=True)
            data = np.clip(data, -threshold, threshold)
            
            return data
            
        except Exception as e:
            logger.error(f"伪迹去除失败: {e}")
            return data
    
    def _standardize(self, data: np.ndarray) -> np.ndarray:
        """标准化处理"""
        try:
            # 按通道标准化
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            
            # 避免除零
            std = np.where(std == 0, 1, std)
            
            standardized_data = (data - mean) / std
            
            return standardized_data
            
        except Exception as e:
            logger.error(f"标准化失败: {e}")
            return data
    
    def _extract_features(self, data: np.ndarray) -> np.ndarray:
        """提取EEG特征
        
        Args:
            data: 预处理后的EEG数据 [C, T]
            
        Returns:
            特征数组 [T_windows, C, D] 
        """
        try:
            # 滑窗特征提取
            window_samples = int(self.window_length * self.sample_rate)
            step_samples = int(window_samples * (1 - self.overlap))
            
            n_channels = data.shape[0]
            n_samples = data.shape[1]
            
            # 计算窗口数量
            n_windows = (n_samples - window_samples) // step_samples + 1
            
            # 特征维度计算
            # 每个频段的功率谱密度 + 相对功率 + 连接性特征
            feature_dim = len(self.frequency_bands) * 3  # PSD + relative power + connectivity
            
            features = np.zeros((n_windows, n_channels, feature_dim))
            
            for i in range(n_windows):
                start = i * step_samples
                end = start + window_samples
                window_data = data[:, start:end]
                
                # 提取窗口特征
                window_features = self._extract_window_features(window_data)
                features[i] = window_features
            
            return features
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            # 返回默认特征
            return self._get_dummy_features().numpy()
    
    def _extract_window_features(self, window_data: np.ndarray) -> np.ndarray:
        """提取单个窗口的特征
        
        Args:
            window_data: 窗口数据 [C, T]
            
        Returns:
            特征数组 [C, D]
        """
        try:
            n_channels = window_data.shape[0]
            feature_dim = len(self.frequency_bands) * 3
            features = np.zeros((n_channels, feature_dim))
            
            for ch_idx in range(n_channels):
                ch_data = window_data[ch_idx]
                
                # 计算功率谱密度
                freqs, psd = signal.welch(
                    ch_data, 
                    fs=self.sample_rate, 
                    nperseg=min(256, len(ch_data)//4),
                    noverlap=None
                )
                
                # 提取各频段特征
                feature_idx = 0
                for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                    # 频段功率
                    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    band_power = np.mean(psd[band_mask]) if np.any(band_mask) else 0
                    
                    features[ch_idx, feature_idx] = band_power
                    feature_idx += 1
                
                # 相对功率
                total_power = np.sum(psd) if np.sum(psd) > 0 else 1
                for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    band_power = np.sum(psd[band_mask])
                    relative_power = band_power / total_power
                    
                    features[ch_idx, feature_idx] = relative_power
                    feature_idx += 1
                
                # 连接性特征 (这里简化为自相关)
                for band_name in self.frequency_bands.keys():
                    autocorr = np.corrcoef(ch_data[:-1], ch_data[1:])[0, 1]
                    if np.isnan(autocorr):
                        autocorr = 0
                    
                    features[ch_idx, feature_idx] = autocorr
                    feature_idx += 1
            
            return features
            
        except Exception as e:
            logger.error(f"窗口特征提取失败: {e}")
            n_channels = window_data.shape[0]
            feature_dim = len(self.frequency_bands) * 3
            return np.zeros((n_channels, feature_dim))
    
    def _resample(self, data: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
        """重采样"""
        try:
            if original_fs == target_fs:
                return data
            
            resampled_data = np.zeros((data.shape[0], int(data.shape[1] * target_fs / original_fs)))
            
            for i in range(data.shape[0]):
                resampled_data[i] = signal.resample(data[i], resampled_data.shape[1])
            
            return resampled_data
            
        except Exception as e:
            logger.error(f"重采样失败: {e}")
            return data
    
    def _get_dummy_features(self) -> torch.Tensor:
        """返回虚拟特征(用于错误处理)"""
        n_windows = 100
        n_channels = len(self.channels)
        feature_dim = len(self.frequency_bands) * 3
        
        return torch.zeros(n_windows, n_channels, feature_dim)
    
    def extract_alpha_power(self, data: np.ndarray) -> float:
        """提取Alpha波功率"""
        try:
            # 全局Alpha波功率
            alpha_powers = []
            
            for ch_idx in range(data.shape[0]):
                ch_data = data[ch_idx]
                freqs, psd = signal.welch(ch_data, fs=self.sample_rate, nperseg=256)
                
                alpha_mask = (freqs >= 8) & (freqs <= 13)
                alpha_power = np.mean(psd[alpha_mask]) if np.any(alpha_mask) else 0
                alpha_powers.append(alpha_power)
            
            return float(np.mean(alpha_powers))
            
        except Exception as e:
            logger.error(f"Alpha波功率提取失败: {e}")
            return 0.0
    
    def extract_theta_beta_ratio(self, data: np.ndarray) -> float:
        """提取Theta/Beta比值"""
        try:
            theta_powers = []
            beta_powers = []
            
            for ch_idx in range(data.shape[0]):
                ch_data = data[ch_idx]
                freqs, psd = signal.welch(ch_data, fs=self.sample_rate, nperseg=256)
                
                theta_mask = (freqs >= 4) & (freqs <= 8)
                beta_mask = (freqs >= 13) & (freqs <= 30)
                
                theta_power = np.mean(psd[theta_mask]) if np.any(theta_mask) else 0
                beta_power = np.mean(psd[beta_mask]) if np.any(beta_mask) else 0
                
                theta_powers.append(theta_power)
                beta_powers.append(beta_power)
            
            avg_theta = np.mean(theta_powers)
            avg_beta = np.mean(beta_powers)
            
            ratio = avg_theta / avg_beta if avg_beta > 0 else 0
            
            return float(ratio)
            
        except Exception as e:
            logger.error(f"Theta/Beta比值提取失败: {e}")
            return 0.0 