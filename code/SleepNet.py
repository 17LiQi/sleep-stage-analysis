from torch import nn


# 混合CNN-LSTM模型
class SleepNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=5):
        super().__init__()

        # CNN特征提取
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, 15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, 11, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(128, 256, 7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
            )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, n_classes)  # 双向LSTM需调整维度
        )
        

    def forward(self, x):
        x = self.features(x)
        x = x.permute(2, 0, 1)
        x, _= self.lstm(x)
        x = x.permute(1, 0, 2)
        return self.classifier(x)
