import torch.nn as nn
import torch.nn.functional as F

from .quant_modules import QuantizedConv2d, QuantizedLinear

class QuantizedBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, bits=8, act_bits=32):
        super(QuantizedBasicBlock, self).__init__()
        self.conv1 = QuantizedConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, bits=bits)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = QuantizedConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, bits=bits)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, bits, act_bits, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.bits = bits
        self.act_bits = act_bits

        self.conv1 = nn.Conv2d(3,64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = QuantizedLinear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, bits=self.bits, act_bits=self.act_bits))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
def build_resnet(bits,num_classes, act_bits=32):
    print(f"Building ResNet with {bits}-bit quantization and {act_bits}-bit activation")
    return ResNet(QuantizedBasicBlock, [2, 2, 2, 2], bits=bits, act_bits=act_bits, num_classes=num_classes)





