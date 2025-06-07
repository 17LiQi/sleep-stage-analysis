import torch
import torch.nn as nn
from models.base_model import BaseModel

class CNNModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        
        # CNN层
        layers = []
        in_channels = 1
        
        for out_channels in config.model.cnn_channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, 
                         kernel_size=config.model.cnn_kernel_size,
                         padding=config.model.cnn_kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 计算CNN输出大小
        self._get_conv_output_size()
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.conv_output_size, config.model.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.hidden_size, config.model.num_classes)
        )
        
    def _get_conv_output_size(self):
        """计算CNN层的输出大小"""
        x = torch.randn(1, 1, self.config.model.input_size)
        x = self.conv_layers(x)
        self.conv_output_size = x.shape[1] * x.shape[2]
        
    def forward(self, x):
        # CNN特征提取
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        
        # 分类
        logits = self.classifier(x)
        
        return logits
    
    def extract_features(self, x):
        """提取特征用于可视化"""
        # CNN特征
        conv_features = self.conv_layers(x)
        conv_features = conv_features.view(conv_features.size(0), -1)
        
        return {
            'conv_features': conv_features
        } 