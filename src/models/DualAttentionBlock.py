import torch
from torch import nn


class DualAttentionBlock(nn.Module):
    """MSDAN论文中的串联双重注意力块 (简化的CBAM)"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool_ch = nn.AdaptiveAvgPool1d(1)
        self.fc_ch = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        self.conv_sp = nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_sp = nn.Sigmoid()

    def forward(self, x):
        ch_attn_weights = self.fc_ch(self.avg_pool_ch(x).squeeze(-1)).unsqueeze(-1)
        x = x * ch_attn_weights

        avg_pool_sp = torch.mean(x, dim=1, keepdim=True)
        max_pool_sp, _ = torch.max(x, dim=1, keepdim=True)
        sp_attn_input = torch.cat((avg_pool_sp, max_pool_sp), dim=1)
        sp_attn_weights = self.sigmoid_sp(self.conv_sp(sp_attn_input))
        x = x * sp_attn_weights
        return x
