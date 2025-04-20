import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """ Focal Loss实现 """

    def __init__(self, weight=None, gamma=3.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        return torch.mean((1 - pt) ** self.gamma * ce_loss)