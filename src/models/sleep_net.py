import torch
import torch.nn as nn
from src.models.base_model import BaseModel

class SleepNet(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        
        # CNN部分
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        # 计算CNN输出大小
        self._get_conv_output_size()
        
        # LSTM部分
        self.lstm = nn.LSTM(
            input_size=self.conv_output_size,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.model.dropout if config.model.num_layers > 1 else 0
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(config.model.hidden_size * 2, 1),
            nn.Tanh()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(config.model.hidden_size * 2, config.model.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.hidden_size, config.model.num_classes)
        )
        
    def _get_conv_output_size(self):
        """计算CNN层的输出大小"""
        x = torch.randn(1, 1, self.config.model.input_size)
        x = self.conv_layers(x)
        self.conv_output_size = x.shape[1]
        
    def forward(self, x):
        # CNN特征提取
        x = self.conv_layers(x)  # [batch, channels, time]
        x = x.transpose(1, 2)    # [batch, time, channels]
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)  # [batch, time, hidden*2]
        
        # 注意力机制
        attention_weights = self.attention(lstm_out)  # [batch, time, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)  # [batch, hidden*2]
        
        # 分类
        logits = self.classifier(context)
        
        return logits
    
    def extract_features(self, x):
        """提取特征用于可视化"""
        # CNN特征
        conv_features = self.conv_layers(x)
        conv_features = conv_features.transpose(1, 2)
        
        # LSTM特征
        lstm_out, _ = self.lstm(conv_features)
        
        # 注意力权重
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        return {
            'conv_features': conv_features,
            'lstm_features': lstm_out,
            'attention_weights': attention_weights
        } 