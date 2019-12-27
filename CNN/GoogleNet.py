"""
@FileName: GoogleNet.py
@Description: Implement GoogleNet
@Author : Ryuk
@CreateDate: 2019/11/14 16:11
@LastEditTime: 2019/11/14 16:11
@LastEditors: Please set LastEditors
@Version: v1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3_1, out_n3x3_2, out_5x5_1, out_5x5_2, pool_size):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3_1, kernel_size=1),
            nn.BatchNorm2d(out_3x3_1),
            nn.ReLU(True),
            nn.Conv2d(out_3x3_1, out_n3x3_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_n3x3_2),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5_1, kernel_size=1),
            nn.BatchNorm2d(out_5x5_1),
            nn.ReLU(True),
            nn.Conv2d(out_5x5_1, out_5x5_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_5x5_2),
            nn.ReLU(True),
            nn.Conv2d(out_5x5_2, out_5x5_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_5x5_2),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_size, kernel_size=1),
            nn.BatchNorm2d(pool_size),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class Googlenet(nn.Module):
    def __init__(self):
        super(Googlenet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

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
    net = Googlenet()
    print(net(x))


if __name__ == '__main__':
    main()

