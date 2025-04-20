import torch
from torch import nn


class SleepNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=5, input_size=3000):
        super(SleepNet, self).__init__()


        # 修改后的特征提取层（减少下采样）
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=50, stride=6, padding=25),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),  # 减小池化窗口

            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),  # 减小步长

            nn.Conv1d(128, 256, kernel_size=8, stride=1, padding=4),  # 增加通道数
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2)
        )

        # 自动计算LSTM输入长度
        self._get_conv_output_dim(input_size)

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=256,  # 与最后一层卷积输出通道一致
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # 优化分类层结构
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # 双向hidden_size*2=256
            nn.GELU(),  # 改用GELU激活
            nn.Dropout(0.4),
            nn.Linear(128, n_classes)
        )

        self.attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')


    def _get_conv_output_dim(self, input_size):
        dummy = torch.zeros(1, 1, input_size, device=next(self.parameters()).device)
        with torch.no_grad():
            out = self.features(dummy)
        self.conv_output_dim = out.size(-1)
        return out.size(-1)



    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.features(x)
        x = x.permute(0, 2, 1)  # [batch, time_steps, features]

        # LSTM处理
        x, _ = self.lstm(x)

        attn_weights = self.attention(x)

        # 使用所有时间步的均值（代替仅用最后一步）
        x = (x * attn_weights).sum(dim=1)

        return self.classifier(x)

    def extract_features(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        features = self.features(x)
        return features.permute(0, 2, 1)  # [batch, timesteps, features]