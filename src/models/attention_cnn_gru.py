# src/models/attention_cnn_gru.py (最终升级版)

import torch
import torch.nn as nn

from src.models.DualAttentionBlock import DualAttentionBlock


class AttentionCNNGRU(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        num_classes = kwargs.get('num_classes', 5)
        gru_hidden_size = kwargs.get('gru_hidden_size', 128)
        gru_num_layers = kwargs.get('gru_num_layers', 2)
        cnn_out_channels = 32

        # --- 【核心修改】从kwargs中获取是否使用注意力的配置 ---
        use_attention = kwargs.get('use_attention', False)

        # 1. CNN编码器 (不变)
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=8, padding=28),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(16, cnn_out_channels, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(cnn_out_channels), nn.ReLU(),
            nn.MaxPool1d(4, stride=4),
        )

        # 2. 【核心修改】“即插即用”的注意力模块
        if use_attention:
            print("AttentionCNNGRU模型已启用双重注意力模块。")
            self.attention = DualAttentionBlock(channels=cnn_out_channels)
        else:
            # 如果不使用，就将其定义为一个“身份”模块
            print("AttentionCNNGRU模型未启用注意力模块。")
            self.attention = nn.Identity()

        # 3. GRU时序建模器 (使用双向)
        self.gru = nn.GRU(
            input_size=cnn_out_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True
        )

        # 4. 最终分类头
        self.fc = nn.Linear(gru_hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (batch, 1, 3000)
        x = self.cnn_encoder(x)  # -> (batch, 32, 11)

        # 【核心修改】直接调用 self.attention
        # 如果是Identity, 它会原封不动地返回x
        # 如果是DualAttentionBlock, 它会返回经过注意力加权的x
        x = self.attention(x)  # -> (batch, 32, 11)

        x = x.permute(0, 2, 1)  # -> (batch, 11, 32)

        output, hidden = self.gru(x)

        last_hidden_forward = hidden[-2, :, :]
        last_hidden_backward = hidden[-1, :, :]
        concat_hidden = torch.cat((last_hidden_forward, last_hidden_backward), dim=1)

        logits = self.fc(concat_hidden)
        return logits