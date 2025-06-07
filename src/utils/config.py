import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    name: str  # 'cnn', 'lstm', 'sleep_net'
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

@dataclass
class DataConfig:
    target_channel: str = "EEG Fpz-Cz"
    window_sec: int = 30
    sampling_rate: int = 100
    dataset_path: str = "/tmp/pycharm_project_506/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
    num_subjects: int = 20
    
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
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    output_dir: str = "/tmp/pycharm_project_506/src/output_results"
    device: str = "cuda"
    seed: int = 42
    
    def __post_init__(self):
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"创建主输出目录: {self.output_dir}")
        # 创建子目录
        for subdir in ['confusion_matrices', 'loss_curves', 'model_checkpoints']:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
        
    @classmethod
    def get_default_config(cls, model_name: str) -> 'Config':
        model_config = ModelConfig(name=model_name)
        return cls(
            model=model_config,
            training=TrainingConfig(),
            data=DataConfig()
        ) 