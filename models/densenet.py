'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module): #bottleneck layers : BN -> ReLU -> Conv
    #in_planes = input dimension  //  growth_rate = output dimension
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__() #다른 클래스의 속성 및 메서드 호출
        self.bn1 = nn.BatchNorm2d(in_planes) #BN
        #1x1 conv를 통해 4*k 개의 feature map 생성
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(4*growth_rate) #BN (4*k) 
        #3x3 conv를 통해 k 개의 feature map으로 줄여 줌
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x): #ReLU
        out = self.conv1(F.relu(self.bn1(x))) #BN 후의 ReLU (1x1 conv 이전)
        out = self.conv2(F.relu(self.bn2(out))) #BN (4*k) 후의 ReLU (3x3 conv 이전)
        out = torch.cat([out,x], 1) #out, x(입력) 텐서를 연결
        return out


class Transition(nn.Module): #transition layer : BN -> 1x1 conv -> 2x2 average pooling
    def __init__(self, in_planes, out_planes): #초기화 메서드
        super(Transition, self).__init__() #다른 클래스의 속성 및 메서드 호출
        self.bn = nn.BatchNorm2d(in_planes) #BN
        #1x1 conv
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x): #2x2 average pooling
        out = self.conv(F.relu(self.bn(x))) #ReLU
        out = F.avg_pool2d(out, 2) #2x2 average pooling layer // 이미지 사이즈를 줄이기 위해
        return out


class DenseNet(nn.Module): #Dense Block, compression
    
    #k = 12, θ = 0.5 
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate #growth_rate = k

        num_planes = 2*growth_rate
        
        #1st conv before Dense Block
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        #1st Dense Block -> Transition Block(1)
        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate #입력 + 레이어 만큼의 growth_rate
        out_planes = int(math.floor(num_planes*reduction)) #feature-map의 수를 줄이기 위해서
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        #2nd Dense Block -> Transition Block(2)
        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        #3rd Dense Block -> Transition Block(3)
        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        #4th Dense Block
        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out)) #1st Dense Block 이후 Transition Layer
        out = self.trans2(self.dense2(out)) #2nd Dense Block 이후 Transition Layer
        out = self.trans3(self.dense3(out)) #3rd Dense Block 이후 Transition Layer
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4) #BN 이후 ReLU
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DenseNet121():
    #[6, 12, 24, 16] = layer 개수 (Block 마다)
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32) #k = 32

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)

def test():
    net = densenet_cifar()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

# test()
