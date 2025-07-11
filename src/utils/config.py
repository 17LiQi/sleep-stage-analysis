import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    name: str  # 'cnn', 'lstm', 'sleep_net', 'wavelet_cnn'
    input_size: int = 3000  # 30秒 * 100Hz
    hidden_size: int = 128
    num_layers: int = 2
    num_classes: int = 5
    dropout: float = 0.5
    
    # CNN特定参数
    cnn_channels: List[int] = None
    cnn_kernel_size: int = 5
    
    # LSTM特定参数
    lstm_bidirectional: bool = True
    
    # 小波CNN特定参数
    wavelet_levels: int = 5
    wavelet_type: str = 'db4'
    
    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 128]

@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    validation_split: float = 0.2
    test_split: float = 0.1
    early_stopping_patience: int = 5
    epochs: int = 50  # 添加缺失的epochs属性

@dataclass
class DataConfig:
    target_channel: str = "EEG Fpz-Cz"
    window_sec: int = 30
    sampling_rate: int = 100
    #dataset_path: str = "/tmp/pycharm_project_506/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
    dataset_path: str = "D:/85970/Files/ProgrammingFiles/GitProject/sleep-stage-analysis/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
    num_subjects: int = 20
    
    # K折交叉验证参数
    n_splits: int = 5  # K折数
    current_fold: int = 0  # 当前折数
    
    # 文件命名规则
    psg_file_pattern: str = "SC{subject_id}E0-PSG.edf"
    hypnogram_file_pattern: str = "SC{subject_id}E{type}-Hypnogram.edf"
    
    # 标签映射
    stage_mapping: Dict[str, int] = field(
        default_factory=lambda: {
            'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3': 3,
            'Sleep stage 4': 3,  # 合并阶段3和4
            'Sleep stage R': 4,
        }
    )

@dataclass
class WaveletConfig:
    """小波变换配置"""
    enabled: bool = False
    wavelet: str = 'db4'  # Daubechies 4小波基
    levels: int = 5  # 分解层数
    mode: str = 'symmetric'  # 边界处理模式
    focus_n1: bool = True  # 是否针对N1阶段优化
    # 频段定义
    freq_bands: Dict[str, tuple] = field(
        default_factory=lambda: {
            'delta': (0.5, 4),    # 慢波睡眠
            'theta': (4, 8),      # N1阶段特征
            'alpha': (8, 14),     # 清醒状态
            'beta': (14, 35),     # 快波
            'gamma': (35, 100)    # 快波
        }
    )

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    wavelet: WaveletConfig = field(default_factory=WaveletConfig)
    #output_dir: str = "/tmp/pycharm_project_506/src/output_results"
    output_dir: str = "D:/85970/Files/ProgrammingFiles/GitProject/sleep-stage-analysis/src/output_results"
    device: str = "cuda"
    seed: int = 42
    
    def __post_init__(self):
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"创建主输出目录: {self.output_dir}")
        # 创建子目录
        for subdir in ['confusion_matrices', 'loss_curves', 'model_checkpoints', 'tsne']:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
        
        # 如果启用小波变换，创建小波分析目录
        if self.wavelet.enabled:
            os.makedirs(os.path.join(self.output_dir, 'wavelet_analysis'), exist_ok=True)
        
    @classmethod
    def get_default_config(cls, model_name: str) -> 'Config':
        model_config = ModelConfig(name=model_name)
        return cls(
            model=model_config,
            training=TrainingConfig(),
            data=DataConfig(),
            wavelet=WaveletConfig()
        )
    
    @classmethod
    def get_wavelet_config(cls, model_name: str, enable_wavelet: bool = True) -> 'Config':
        """获取启用小波变换的配置"""
        model_config = ModelConfig(name=model_name)
        wavelet_config = WaveletConfig(enabled=enable_wavelet)
        
        return cls(
            model=model_config,
            training=TrainingConfig(),
            data=DataConfig(),
            wavelet=wavelet_config
        ) 