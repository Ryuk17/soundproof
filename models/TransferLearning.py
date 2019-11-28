# -*- coding: UTF-8 -*-
"""
@FileName: TransferLearning.py
@Description: Implement TransferLearning
@Author: Lj
@CreateDate: 2019/11/28 14:50
@LastEditTime: 2019/11/28 14:50
@LastEditors: Please set LastEditors
@Version: v1.0
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50

class TransferNet(nn.Module):
    def __init__(self, model, input_dim, output_dim):
        super(TransferNet, self).__init__()

        self.pre_layers = nn.Sequential(*list(model.children()))[:-1]
        self.last_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.pre_layers(x)
        x = x.view(x.size(0), -1)
        x = self.last_layer(x)
        return x

def set_optimizer(model, lr_base, momentum, w_decay):
    last_params = map(id, model.last_layer.parameters())
    pre_params = filter(lambda addr: id(addr) not in last_params, model.parameters())
    optimizer = torch.optim.SGD([
        {'params': pre_params},
        {'params': model.last_layer.parameters(), 'lr': 0.1}], lr=lr_base, momentum = momentum, weight_decay=w_decay)

    return optimizer

def main():
    model = resnet50()
    model = TransferNet(model, 2048, 100)
    x = torch.randn(1,3,224,224)
    print(model(x))


if __name__ == '__main__':
    main()
