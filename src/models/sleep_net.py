import torch
import torch.nn as nn
from models.base_model import BaseModel


class SleepNet(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        # 改进的双臂 CNN
        self.conv_small = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=25, stride=1, padding=12),  # 减小滤波器大小
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=25, stride=1, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=25, stride=1, padding=12),  # 增加一层
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.conv_large = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=200, stride=1, padding=100),  # 减小滤波器大小
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=200, stride=1, padding=100),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=200, stride=1, padding=100),  # 增加一层
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # 计算 CNN 输出大小
        self._get_conv_output_size()

        # 改进的 Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=self.conv_output_size,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.model.dropout if config.model.num_layers > 1 else 0
        )

        # 改进的多头自注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=config.model.hidden_size * 2,
            num_heads=16,  # 增加注意力头数
            dropout=config.model.dropout
        )

        # 改进的分类器
        self.classifier = nn.Sequential(
            nn.Linear(config.model.hidden_size * 2, config.model.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.hidden_size, config.model.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.hidden_size // 2, config.model.num_classes)
        )

    def _get_conv_output_size(self):
        """计算双臂 CNN 的输出大小"""
        x = torch.randn(1, 1, self.config.model.input_size)
        x_small = self.conv_small(x)
        x_large = self.conv_large(x)
        x = torch.cat([x_small, x_large], dim=1)
        self.conv_output_size = x.shape[1]

    def forward(self, x):
        # 双臂 CNN 特征提取
        x_small = self.conv_small(x)
        x_large = self.conv_large(x)
        x = torch.cat([x_small, x_large], dim=1)
        x = x.transpose(1, 2)

        # Bi-LSTM 处理
        lstm_out, _ = self.lstm(x)

        # 多头自注意力
        lstm_out = lstm_out.transpose(0, 1)
        attn_output, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 使用注意力加权平均
        context = torch.mean(attn_output, dim=0)  # 使用平均池化而不是只取最后一个时间步

        # 分类
        logits = self.classifier(context)

        return logits

    def extract_features(self, x):
        """提取特征用于可视化"""
        # 双臂 CNN 特征
        x_small = self.conv_small(x)
        x_large = self.conv_large(x)
        conv_features = torch.cat([x_small, x_large], dim=1)
        conv_features = conv_features.transpose(1, 2)

        # Bi-LSTM 特征
        lstm_out, _ = self.lstm(conv_features)

        # 注意力权重
        lstm_out_attn = lstm_out.transpose(0, 1)
        _, attn_weights = self.attention(lstm_out_attn, lstm_out_attn, lstm_out_attn)

        return {
            'conv_features': conv_features,
            'lstm_features': lstm_out,
            'attention_weights': attn_weights
        }