"""
@ Filename:       RecurrentDenoisingAutoencoder.py
@ Author:         Danc1elion
@ Create Date:    2020-01-08   
@ Update Date:    2020-01-08 
@ Description:    Implement RecurrentDenoisingAutoencoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RDAE(nn.Module):
    def __init__(self):
        super(RDAE).__init__()

    def adjustInput(self, x):
        feature_dim = x.shape[2]                # batch_size, time_step, feature
        padding = torch.zeros(feature_dim)
        x = torch.cat((padding, x), dim=1)
        x = torch.cat((x, padding), dim=1)
        return x

    def encoder(self, x):
        pass

    def decoder(self, x):
        pass

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

def main():
    x = torch.randn(1, 1, 28, 28)
    net = RDAE()
    print(net(x))


if __name__ == '__main__':
    main()






