import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from ..quant_modules import QuantizedConv2d

class Swish(nn.Module):
    def forward(self, x): 
        return x * torch.sigmoid(x)
    
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Conv2d(in_channels, reduced_dim, 1), Swish(),
            nn.Conv2d(reduced_dim, in_channels, 1), nn.Sigmoid(),
        )
    def forward(self, x): 
        return x*self.se(x)
        

class QuantizedMBConv(nn.Module):
    def __init__(self, in_c, out_c, k, s, e_r, se_r, bits, act_bits):
        super().__init__()
        hidden_c = in_c * e_r
        self.use_res = (s == 1 and in_c == out_c)
        reduced_dim = max(1, int(in_c * se_r))

        layers = []
        if e_r != 1:
            layers.extend([QuantizedConv2d(in_c, hidden_c, 1, bias = False, bits=bits), nn.BatchNorm2d(hidden_c), Swish()])

        layers.extend([QuantizedConv2d(hidden_c, hidden_c, k, s, padding=(k-1)//2, groups=hidden_c, bias =False, bits=bits), nn.BatchNorm2d(hidden_c), Swish()])
        layers.append(SqueezeExcitation(hidden_c, reduced_dim))
        layers.extend([QuantizedConv2d(hidden_c, out_c, 1, bias=False, bits=bits), nn.BatchNorm2d(out_c)])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)
    

class EfficientNet(nn.Module):
    def __init__(self, bits, act_bits, num_classes):
        super().__init__()
        settings = [
            (1, 16, 1, 1, 3), (6, 24, 2, 2, 3), (6, 40, 2, 2, 5),
            (6, 80, 3, 2, 3), (6, 112, 3, 1, 5), (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3)
        ]

        in_c = 32
        self.stem = nn.Sequential(nn.Conv2d(3, in_c, 3, 2, 1, bias=False), nn.BatchNorm2d(in_c), Swish())

        layers = []
        for t, c, n, s, k in settings:
            for i in range(n):
                stride = s if i ==0 else 1
                layers.append(QuantizedMBConv(in_c, c, k, stride, t, 0.25, bits, act_bits))
                in_c = c
        self.layers = nn.Sequential(*layers)

        self.head_conv = nn.Sequential(
            nn.Conv2d(in_c, 1280, 1, bias=False), 
            nn.BatchNorm2d(1280), 
            Swish(),
            nn.AdaptiveAvgPool2d(1) # 여기까지는 4D 텐서를 다룸
        )
        
        # Linear 레이어는 따로 정의
        self.head_linear = nn.Linear(1280, num_classes)
        
        # Dropout은 forward에서 직접 적용하는 것이 더 유연함 (선택사항)
        self.dropout_rate = 0.2 # B0의 표준 드롭아웃 비율

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.head_conv(x)
        x = x.view(x.size(0), -1)
        # Dropout은 학습 시에만 적용
        if self.training:
            x = F.dropout(x, p=self.dropout_rate)
            
        x = self.head_linear(x)
        return x

def build_efficientnet(bits, num_classes, act_bits=32):
    return EfficientNet(bits, act_bits, num_classes)