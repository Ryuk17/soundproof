"""
@ Filename:       DenoisingAutoencoder.py
@ Author:         Danc1elion
@ Create Date:    2020-01-08   
@ Update Date:    2020-01-08 
@ Description:    Implement DenoisingAutoencoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DAE(nn.Module):
    def __init__(self):
        super(DAE, self).__init__()

    def corruption(self, x, factor=0.2):
        noise = factor * torch.rand(x.shape)
        x = x + noise
        x = torch.clamp(x, 0., 1.)
        return x

    def encoder(self, x):
        x = self.corruption(x)
        features = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # (b, 16, 10, 10)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # (b, 16, 5, 5)
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # (b, 8, 3, 3)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # (b, 8, 2, 2)
        )
        return features(x)

    def decoder(self, x):
        features = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # (b, 16, 5, 5)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # (b, 8, 15, 15)
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # (b, 1, 28, 28)
            nn.Tanh()
        )
        return features(x)

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


def main():
    x = torch.randn(1, 1, 28, 28)
    net = DAE()
    print(net(x))


if __name__ == '__main__':
    main()