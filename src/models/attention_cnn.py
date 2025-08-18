# src/models/attention_cnn.py (改造后)
import torch.nn as nn
from .attention_cnn_gru import DualAttentionBlock


class AttentionCNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        num_classes = kwargs.get('num_classes', 5)
        use_attention = kwargs.get('use_attention', False)  # 默认不启用

        # 1. CNN特征提取器 (提取特征序列)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=8, padding=28),
            nn.ReLU(),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(16, 32, kernel_size=8, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=4)  # 输出: (batch, 32, 11)
        )

        cnn_out_channels = 32

        # 2. 【新增】可选的注意力模块
        if use_attention:
            print("AttentionCNN模型已启用双重注意力模块。")
            self.attention = DualAttentionBlock(channels=cnn_out_channels)
        else:
            self.attention = nn.Identity()

        # 3. 聚合与分类
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cnn_out_channels, num_classes)

    def forward(self, x):
        # 提取特征序列
        x = self.feature_extractor(x)

        # (可选地) 应用注意力
        x = self.attention(x)

        # 聚合特征并分类
        x = self.pooling(x).squeeze(-1)
        logits = self.fc(x)
        return logits