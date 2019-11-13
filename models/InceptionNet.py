"""
@FileName: InceptionNet.py
@Description: Implement InceptionNet
@Author : Lj
@CreateDate: 2019/11/13 16:35
@LastEditTime: 2019/11/13 16:35
@LastEditors: Please set LastEditors
@Version: v1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class basicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(basicBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, *args, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x, *args, **kwargs):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(Inception, self).__init__()

        self.branch1x1 = basicBlock(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = basicBlock(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = basicBlock(48, 64, kernel_size=5, padding=2)

        self.branch3x3_1 = basicBlock(in_channels, 64, kernel_size=1)
        self.branch3x3_2 = basicBlock(64, 96, kernel_size=3, padding=1)
        self.branch3x3_3 = basicBlock(96, 96, kernel_size=3, padding=1)

        self.branch_pool = basicBlock(in_channels, pool_features, kernel_size=1)


    def forward(self, x, *args, **kwargs):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        x = torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], 1)
        return x

class Inceptionnet(nn.Module):
    def __init__(self):
        super(Inceptionnet, self).__init__()

        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        self.a3 = Inception(192, 32)
        self.b3 = Inception(256, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 64)
        self.b4 = Inception(512, 64)
        self.c4 = Inception(512, 64)
        self.d4 = Inception(512, 64)
        self.e4 = Inception(528, 128)

        self.a5 = Inception(832, 128)
        self.b5 = Inception(832, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.maxpool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def main():
    x = torch.randn(1, 3, 32, 32)
    net = Inceptionnet()

    print(net(x))


if __name__ == '__main__':
    main()
