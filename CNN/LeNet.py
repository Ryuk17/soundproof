# -*- coding: utf-8 -*-
# @FileName: LeNet.py
# @Description: Implementation of LeNet
# @Author  : Ryuk
# @CreateDate: 2019/11/4 17:25
# @LastEditTime: 2019/11/4 17:25
# @LastEditors: Please set LastEditors
# @Version:

import torch
import torch.nn as nn

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(3, 6, 5, padding=1))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(6, 16, 5))
        layer2.add_module('pool2', nn.MaxPool2d(2, 2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(400, 120))
        layer3.add_module('fc2', nn.Linear(120, 84))
        layer3.add_module('fc3', nn.Linear(84, 10))
        self.layer3 = layer3


    def forward(self, x):
        assert x.size(-1) == 32
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x


def main():
    net = Lenet()
    x = torch.randn(1, 3, 32, 32)
    print(net(x))

if __name__ == '__main__':
    main()