# this file is part of the Satellite Efficient project

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, bits):
        if bits <= 16:
            return tensor
        
        if bits == 1:
            alpha = tensor.abs().mean()
            return alpha * tensor.sign()
        
        if bits == 2:
            alpha2 = tensor.abs().mean()
            delta = 0.7 * alpha2
            output = torch.zeros_like(tensor)
            output[tensor < delta] = 1.0
            output[tensor > -delta] = -1.0
            return alpha2 * output
        
        # q_min, q_max = 0., 2.**bits -1.
        # t_min, t_max = tensor.min(), tensor.max()
        # scale = (t_max - t_min) / (q_max - q_min)
        # if scale < 1e-10: return tensor
        # zero_point = torch.round(q_min - t_min / scale)
        # q_tensor = torch.round(tensor / scale + zero_point).clamp(q_min, q_max)
        # deq_tensor = (q_tensor - zero_point) * scale
        # return deq_tensor
        # q_min, q_max를 음수 포함하도록 변경
        q_min = -2.**(bits - 1)
        q_max = 2.**(bits - 1) - 1
        
        # 1. 스케일(scale) 계산 (min/max 대신 abs().max() 사용)
        abs_max = tensor.abs().max()
        scale = abs_max / q_max
        if scale < 1e-10: return tensor

        # 2. 양자화 및 역양자화 (zero_point 불필요)
        q_tensor = torch.round(tensor / scale).clamp(q_min, q_max)
        deq_tensor = q_tensor * scale
        
        return deq_tensor
    

    @staticmethod
    def backward(ctx, grad_output):
        # STE: Straight-Through Estimator
        # Gradient is passed through unchanged
        return grad_output, None
    

class QuantizedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, bits=8):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)
        self.bits = bits
        if self.bits <= 4:
            self.weight_fp = nn.Parameter(self.weight.data.clone())
    def forward(self, x):
        if self.bits >=16:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # if/else 문을 bits 값에 따라 분리하여, weight_fp가 없을 때 참조하지 않도록 함
        if self.bits <= 4:
            # 1, 2, 4-bit는 안정적인 weight_fp를 양자화
            quantized_weight = QuantizeSTE.apply(self.weight_fp, self.bits)
        else: # 8-bit
            # 8-bit는 QAT처럼 원본 weight를 직접 양자화 시뮬레이션
            quantized_weight = QuantizeSTE.apply(self.weight, self.bits)
        return F.conv2d(x, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuantizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bits=8):
        super().__init__(in_features, out_features, bias=bias)
        self.bits = bits
        if self.bits <=4:
            self.weight_fp = nn.Parameter(self.weight.data.clone())

    def forward(self, x):
        if self.bits >= 16:
            return F.linear(x, self.weight, self.bias)

        if self.bits <= 4:
            quantized_weight = QuantizeSTE.apply(self.weight_fp, self.bits)
        else:
            quantized_weight = QuantizeSTE.apply(self.weight, self.bits)

        return F.linear(x, quantized_weight, self.bias)

def clip_weights(model, min_val=1.0, max_val=1.0):
    for module in model.modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLinear)) and module.bits <= 4:
            if hasattr(module, 'weight_fp'):
                module.weight_fp.data.clamp_(min_val, max_val)

