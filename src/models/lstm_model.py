import torch
import torch.nn as nn
from models.base_model import BaseModel

class LSTMModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        
        # 输入投影层
        self.input_proj = nn.Linear(1, config.model.hidden_size)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=config.model.hidden_size,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            batch_first=True,
            bidirectional=config.model.lstm_bidirectional,
            dropout=config.model.dropout if config.model.num_layers > 1 else 0
        )
        
        # 注意力机制
        lstm_output_size = config.model.hidden_size * 2 if config.model.lstm_bidirectional else config.model.hidden_size
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, 1),
            nn.Tanh()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, config.model.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.hidden_size, config.model.num_classes)
        )
        
    def forward(self, x):
        # 输入投影 [batch, channels, time] -> [batch, time, hidden]
        x = x.transpose(1, 2)  # [batch, time, channels]
        x = self.input_proj(x)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 注意力机制
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 分类
        logits = self.classifier(context)
        
        return logits
    
    def extract_features(self, x):
        """提取特征用于可视化"""
        # 输入投影
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        
        # LSTM特征
        lstm_out, _ = self.lstm(x)
        
        # 注意力权重
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        return {
            'lstm_features': lstm_out,
            'attention_weights': attention_weights
        } 