from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, x):
        """前向传播"""
        pass
    
    @abstractmethod
    def extract_features(self, x):
        """特征提取，用于可视化等"""
        pass
    
    def get_optimizer(self):
        """获取优化器"""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
    
    def get_scheduler(self, optimizer):
        """获取学习率调度器"""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.1,
            patience=5,
            verbose=True
        ) 