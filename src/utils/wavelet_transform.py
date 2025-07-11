import numpy as np
import torch
import torch.nn as nn
import pywt
from typing import Tuple, List, Optional
import logging
import sys
import os

# 添加src目录到路径以导入BaseModel
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.base_model import BaseModel

logger = logging.getLogger(__name__)

class WaveletTransform:
    """
    小波变换模块，专门针对睡眠阶段分类，特别是N1阶段
    使用Daubechies 4小波基，5层分解覆盖所有脑电频段
    """

    def __init__(self,
                 wavelet='db4',
                 levels=5,
                 mode='symmetric',
                 focus_n1: bool = True):
        """
        初始化小波变换

        Args:
            wavelet: 小波基函数，默认db4适合生物信号
            levels: 分解层数，5层覆盖delta到gamma波段
            mode: 边界处理模式
            focus_n1: 是否针对N1阶段优化
        """
        self.wavelet = wavelet
        self.levels = levels
        self.mode = mode
        self.focus_n1 = focus_n1

        # 脑电频段定义
        self.freq_bands = {
            'delta': (0.5, 4),    # 慢波睡眠
            'theta': (4, 8),      # N1阶段特征
            'alpha': (8, 14),     # 清醒状态
            'beta': (14, 35),     # 快波
            'gamma': (35, 100)    # 快波
        }

        logger.info(f"初始化小波变换: {wavelet}, {levels}层分解")
        if focus_n1:
            logger.info("启用N1阶段优化模式")

    def decompose_signal(self, signal: np.ndarray, sampling_rate: int = 100) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        对信号进行小波分解

        Args:
            signal: 输入信号 (1D数组)
            sampling_rate: 采样率，默认100Hz

        Returns:
            coeffs: 小波系数列表 [cA, cD1, cD2, cD3, cD4, cD5]
        """
        try:
            # 进行小波分解
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.levels, mode=self.mode)

            # 计算每个分解层对应的频段
            freq_bands = self._get_frequency_bands(sampling_rate)

            logger.debug(f"小波分解完成，系数数量: {len(coeffs)}")
            for i, (coeff, freq_range) in enumerate(zip(coeffs, freq_bands)):
                logger.debug(f"层 {i}: 频段 {freq_range[0]:.1f}-{freq_range[1]:.1f}Hz, 系数数量: {len(coeff)}")

            return coeffs

        except Exception as e:
            logger.error(f"小波分解失败: {e}")
            raise

    def _get_frequency_bands(self, sampling_rate: int) -> List[Tuple[float, float]]:
        """
        计算每个分解层对应的频段

        Args:
            sampling_rate: 采样率

        Returns:
            频段列表 [(低频, 高频), ...]
        """
        nyquist = sampling_rate / 2
        freq_bands = []

        for i in range(self.levels + 1):
            if i == 0:  # 近似系数
                low_freq = 0
                high_freq = nyquist / (2 ** self.levels)
            else:  # 细节系数
                low_freq = nyquist / (2 ** i)
                high_freq = nyquist / (2 ** (i - 1))

            freq_bands.append((low_freq, high_freq))

        return freq_bands

    def extract_features(self, coeffs: List[np.ndarray]) -> np.ndarray:
        """
        从小波系数中提取特征

        Args:
            coeffs: 小波系数列表

        Returns:
            特征向量
        """
        features = []

        for i, coeff in enumerate(coeffs):
            # 统计特征
            mean_val = np.mean(coeff)
            std_val = np.std(coeff)
            energy = np.sum(coeff ** 2)
            entropy = -np.sum(coeff ** 2 * np.log(np.abs(coeff) + 1e-10))

            # 针对N1阶段的特殊特征
            if self.focus_n1 and i in [2, 3]:  # theta和alpha波段
                # 计算功率谱密度
                power_spectrum = np.abs(np.fft.fft(coeff)) ** 2

                # 安全检查：确保切片不为空
                half_len = len(power_spectrum) // 2
                if half_len > 1:
                    power_slice = power_spectrum[1:half_len]
                    if len(power_slice) > 0:
                        dominant_freq_idx = np.argmax(power_slice) + 1
                        dominant_freq = dominant_freq_idx * 100 / len(coeff)  # 假设采样率100Hz

                        # 计算频带功率比
                        total_power = np.sum(power_spectrum)
                        if total_power > 0:
                            power_ratio = power_spectrum[dominant_freq_idx] / total_power
                        else:
                            power_ratio = 0
                    else:
                        dominant_freq = 0
                        power_ratio = 0
                else:
                    dominant_freq = 0
                    power_ratio = 0

                features.extend([mean_val, std_val, energy, entropy, dominant_freq, power_ratio])
            else:
                features.extend([mean_val, std_val, energy, entropy])

        return np.array(features)

    def reconstruct_band(self, coeffs: List[np.ndarray], band_level: int) -> np.ndarray:
        """
        重构特定频段的信号

        Args:
            coeffs: 小波系数
            band_level: 要重构的频段层级 (1-5)

        Returns:
            重构的信号
        """
        if band_level < 1 or band_level > self.levels:
            raise ValueError(f"频段层级必须在1到{self.levels}之间")

        # 创建新的系数列表，只保留指定频段
        new_coeffs = [np.zeros_like(coeff) for coeff in coeffs]
        new_coeffs[band_level] = coeffs[band_level]  # 保留指定频段

        # 重构信号
        reconstructed = pywt.waverec(new_coeffs, self.wavelet, mode=self.mode)

        return reconstructed

    def get_n1_enhanced_features(self, coeffs: List[np.ndarray]) -> np.ndarray:
        """
        针对N1阶段增强的特征提取

        Args:
            coeffs: 小波系数

        Returns:
            增强的特征向量
        """
        features = []

        # 基础特征
        base_features = self.extract_features(coeffs)
        features.extend(base_features)

        # N1阶段特殊特征
        if self.focus_n1:
            # theta波段特征 (第3层，4-8Hz)
            theta_coeff = coeffs[3] if len(coeffs) > 3 else np.zeros(1)
            theta_energy = np.sum(theta_coeff ** 2)
            theta_std = np.std(theta_coeff)

            # alpha波段特征 (第2层，8-14Hz)
            alpha_coeff = coeffs[2] if len(coeffs) > 2 else np.zeros(1)
            alpha_energy = np.sum(alpha_coeff ** 2)
            alpha_std = np.std(alpha_coeff)

            # theta/alpha比率 (N1阶段的重要指标)
            if alpha_energy > 0:
                theta_alpha_ratio = theta_energy / alpha_energy
            else:
                theta_alpha_ratio = 0

            # 频段间能量分布
            total_energy = sum(np.sum(coeff ** 2) for coeff in coeffs[1:])
            if total_energy > 0:
                theta_energy_ratio = theta_energy / total_energy
                alpha_energy_ratio = alpha_energy / total_energy
            else:
                theta_energy_ratio = alpha_energy_ratio = 0

            # 添加N1特定特征
            n1_features = [
                theta_energy, theta_std, alpha_energy, alpha_std,
                theta_alpha_ratio, theta_energy_ratio, alpha_energy_ratio
            ]
            features.extend(n1_features)

        return np.array(features)

    def get_feature_dim(self, signal_len: int = 3000, sampling_rate: int = 100) -> int:
        """
        获取特征维度（根据输入信号长度和采样率动态计算）
        """
        dummy = np.zeros(signal_len)
        coeffs = self.decompose_signal(dummy, sampling_rate)
        features = self.get_n1_enhanced_features(coeffs)
        return features.shape[0]


class WaveletCNN(BaseModel):
    """
    结合小波变换的CNN模型
    """

    def __init__(self, config):
        super().__init__(config)
        self.wavelet_transform = WaveletTransform(levels=config.model.wavelet_levels)
        # 动态获取特征维度
        wavelet_feature_dim = self.wavelet_transform.get_feature_dim(signal_len=config.model.input_size)
        # 原始信号特征
        self.signal_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        # 小波特征处理
        self.wavelet_fc = nn.Sequential(
            nn.Linear(wavelet_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.model.dropout)
        )
        # 特征融合
        self.fusion_fc = nn.Sequential(
            nn.Linear(128 + 64, 128),  # 128(信号) + 64(小波)
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(128, config.model.num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        # 处理原始信号
        if x.dim() == 2:
            x_signal = x.unsqueeze(1)  # [batch, 1, length]
        elif x.dim() == 3:
            x_signal = x  # [batch, 1, length]
        else:
            raise ValueError(f"输入信号维度不正确: {x.shape}")
        signal_features = self.signal_conv(x_signal).squeeze(-1)  # [batch, 128]
        # 处理小波特征
        wavelet_features = []
        for i in range(batch_size):
            signal = x[i].cpu().numpy()
            coeffs = self.wavelet_transform.decompose_signal(signal)
            features = self.wavelet_transform.get_n1_enhanced_features(coeffs)
            wavelet_features.append(features)
        wavelet_features = torch.tensor(np.array(wavelet_features),
                                      dtype=torch.float32,
                                      device=x.device)
        wavelet_features = self.wavelet_fc(wavelet_features)  # [batch, 64]
        # 特征融合
        combined_features = torch.cat([signal_features, wavelet_features], dim=1)
        output = self.fusion_fc(combined_features)
        return output

    def extract_features(self, x):
        """提取特征用于可视化"""
        batch_size = x.size(0)
        # 处理原始信号
        if x.dim() == 2:
            x_signal = x.unsqueeze(1)  # [batch, 1, length]
        elif x.dim() == 3:
            x_signal = x  # [batch, 1, length]
        else:
            raise ValueError(f"输入信号维度不正确: {x.shape}")
        signal_features = self.signal_conv(x_signal).squeeze(-1)

        # 处理小波特征
        wavelet_features = []
        for i in range(batch_size):
            signal = x[i].cpu().numpy()
            coeffs = self.wavelet_transform.decompose_signal(signal)
            features = self.wavelet_transform.get_n1_enhanced_features(coeffs)
            wavelet_features.append(features)
        wavelet_features = torch.tensor(np.array(wavelet_features),
                                      dtype=torch.float32,
                                      device=x.device)
        wavelet_features = self.wavelet_fc(wavelet_features)

        return {
            'signal_features': signal_features,
            'wavelet_features': wavelet_features,
            'combined_features': torch.cat([signal_features, wavelet_features], dim=1)
        }


class WaveletPreprocessor:
    """
    小波预处理模块，用于数据预处理
    """

    def __init__(self, wavelet_transform: WaveletTransform):
        self.wavelet_transform = wavelet_transform

    def process_batch(self, signals: np.ndarray) -> np.ndarray:
        """
        批量处理信号

        Args:
            signals: 信号数组 [batch_size, signal_length]

        Returns:
            处理后的特征数组 [batch_size, feature_dim]
        """
        features = []

        for signal in signals:
            coeffs = self.wavelet_transform.decompose_signal(signal)
            signal_features = self.wavelet_transform.get_n1_enhanced_features(coeffs)
            features.append(signal_features)

        return np.array(features)

    def process_single(self, signal: np.ndarray) -> np.ndarray:
        """
        处理单个信号

        Args:
            signal: 单个信号

        Returns:
            特征向量
        """
        coeffs = self.wavelet_transform.decompose_signal(signal)
        return self.wavelet_transform.get_n1_enhanced_features(coeffs)