from d2l.torch import PositionalEncoding
from torch import nn


class SleepTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=8, n_classes=5):
        super().__init__()

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=4
        )

        self.embed = nn.Sequential(
            nn.Conv1d(1, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model)
        )

        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(2, 0, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=0)
        x = self.classifier(x)

