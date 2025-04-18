from torch import nn
import torch


# 混合CNN-LSTM模型
class SleepNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=5, input_size=3000):
        super(SleepNet, self).__init__()
        
        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=50, stride=6, padding=25),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )

        # 计算卷积层输出大小

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=128,  # 卷积层输出通道数
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(256, n_classes),  # 双向LSTM，所以是hidden_size * 2
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(n_classes, 5)
        )
        
    def _get_conv_output_size(self, input_size):
        # 模拟卷积层的前向传播来计算输出大小
        x = torch.zeros(1, 1, input_size)
        x = self.features(x)
        return x.size(2)
    
    def forward(self, x):
        # 确保输入维度正确 [batch_size, channels, length]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 添加通道维度
        
        # 特征提取
        x = self.features(x)
        
        # 调整维度以适应LSTM [batch_size, seq_len, features]
        x = x.permute(0, 2, 1)
        
        # LSTM层
        x, _ = self.lstm(x)
        
        # 只使用最后一个时间步的输出
        x = x[:, -1, :]
        
        # 分类
        x = self.classifier(x)
        
        return x
