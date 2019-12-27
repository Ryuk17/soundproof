# -*- coding: UTF-8 -*-
"""
@FileName: MobileNetV1.py
@Description: Implement MobileNetV1
@Author: Ryuk
@CreateDate: 2019/12/9 13:35
@LastEditTime: 2019/12/9 13:35
@LastEditors: Please set LastEditors
@Version: v1.0
"""

import time
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True))

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True))

        self.feature = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7) )

        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



def main():
    x = torch.randn(1, 3, 224, 224)
    net = MobileNet()
    print(net(x))


if __name__ == '__main__':
    main()