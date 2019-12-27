"""
@FileName: VGG.py
@Description: Implementation of VGG
@Author : Ryuk
@CreateDate: 2019/11/13 15:44
@LastEditTime: 2019/11/13 15:44
@LastEditors: Please set LastEditors
@Version: v1.0
"""

import torch
import torch.nn as nn


class VGGNet(nn.Module):
    def __init__(self, n_classes, n_layers, in_channels=3):
        super(VGGNet, self).__init__()

        self.architecture = {
            16: [64, 64, 'Pooling', 128, 128, 'Pooling', 256, 256, 256, 'Pooling', 512, 512, 512, 'Pooling', 512, 512, 512, 'Pooling'],
            19: [64, 64, 'Pooling', 128, 128, 'Pooling', 256, 256, 256, 256, 'Pooling', 512, 512, 512, 512, 'Pooling', 512, 512, 512, 512, 'Pooling'],
        }
        self.features = self._constructNet(self.architecture[n_layers], in_channels)
        self.classifier = nn.Linear(512, n_classes)

    def _constructNet(self, arch, in_channels):
        net = []
        for out_channels in arch:
            if out_channels == 'Pooling':
                net += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                net += [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ]
                in_channels = out_channels
        net += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*net)

    def forward(self, x):
        assert x.size(-1) == 32
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def main():
    x = torch.randn(1, 3, 32, 32)
    net = VGGNet(10, 19)
    print(net(x))


if __name__ == '__main__':
    main()
