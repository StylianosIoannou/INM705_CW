import torch
import torch.nn as nn

# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Global average and max pooling reduce spatial dimensions to 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP: two conv layers with 1x1 kernels and ReLU in between
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)

        # Sigmoid to normalize attention weights
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Generate channel-wise attention from both pooled features
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        # Multiply attention weights with input feature map
        return x * self.sigmoid(out)

# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Use large kernel for capturing spatial context
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute channel-wise average and max projections
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate and convolve to get spatial attention map
        concat = torch.cat((avg_pool, max_pool), dim=1)
        return x * self.sigmoid(self.conv(concat))

# CBAM Module (Channel + Spatial Attention)
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Initialize channel and spatial attention blocks
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply channel attention first, then spatial attention
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
