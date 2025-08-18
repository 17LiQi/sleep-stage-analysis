# src/models/attention_resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock1D(nn.Module):
    """标准的1D ResNet基础块"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        # 如果维度或步长发生变化，需要调整shortcut连接以匹配维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AttentionResNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        num_classes = kwargs.get('num_classes', 5)
        self.is_encoder_only = kwargs.get('is_encoder_only', False)
        use_attention = kwargs.get('use_attention', True)
        
        self.in_planes = 64

        # 初始卷积层，大幅降低时间维度
        self.conv1 = nn.Conv1d(1, 64, kernel_size=63, stride=8, padding=28, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(BasicBlock1D, 64, 2, stride=2)
        self.layer2 = self._make_layer(BasicBlock1D, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock1D, 256, 2, stride=2)
        # 编码器的输出特征维度是 256
        
        # 定义完整的编码器部分
        self.encoder = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(),
            self.layer1, self.layer2, self.layer3
        )

        # 注意力层 (只在微调时使用)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
                nn.Softmax(dim=1)
            )
            # 分类头 (只在微调时使用)
            self.classifier = nn.Linear(256, num_classes)
        else:
            # 如果不使用，就用平均池化作为默认
            self.attention = lambda x: x.mean(dim=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch, 1, 3000)
        encoded = self.encoder(x) # shape: (batch, 256, seq_len_after_convs)
        
        if self.is_encoder_only:
            # 【关键修复】在预训练时，直接返回三维的编码器输出
            return encoded

        # --- 微调逻辑 ---
        # 调整维度以应用注意力
        encoded_permuted = encoded.permute(0, 2, 1) # (batch, seq_len, 256)
        
        # 计算注意力权重
        attn_weights = self.attention(encoded_permuted) # (batch, seq_len, 1)
        
        # 使用注意力权重对编码器输出进行加权求和
        context_vector = torch.sum(attn_weights * encoded_permuted, dim=1) # (batch, 256)
        
        # 分类
        logits = self.classifier(context_vector)
        return logits

    def load_encoder(self, path):
        """辅助函数，用于从预训练文件加载编码器权重"""
        try:
            # 预训练时保存的是整个 AttentionResNet (is_encoder_only=True) 的 state_dict
            # 它的所有层都在 encoder 属性中
            pretrained_dict = torch.load(path, map_location='cpu', weights_only=True)
            model_dict = self.encoder.state_dict()
            
            # 1. 过滤掉不匹配的键 (如果存在)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            
            # 2. 更新当前模型的 state_dict
            model_dict.update(pretrained_dict) 
            
            # 3. 加载构建的 state_dict
            self.encoder.load_state_dict(model_dict)
            
            print(f"成功从 {path} 加载预训练的编码器权重。")
        except Exception as e:
            print(f"加载预训练编码器失败: {e}")