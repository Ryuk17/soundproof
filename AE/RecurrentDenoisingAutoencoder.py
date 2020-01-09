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
        self.batch_size = 0
        self.seq_len = 0
        self.feature_dim = 0

    def adjustInput(self, x):
        # get batch_size, seq_len, feature
        self.batch_size, self.seq_len, self.feature_dim = x.shape[0], x.shape[1], x.shape[2]

        # adjust shape into seq_len, batch_size, feature
        x = torch.transpose(x, 0, 1)

        # pre-padding and post-padding
        padding = torch.zeros(x[0].shape)
        x = torch.cat((padding, x, padding), dim=0)

        # enframe x into 3 frame combination

        return x

    def encoder(self, x):
        x = self.adjustInput(x)
        # first fc layer
        features = nn.Sequential(nn.Linear(self.feature_dim, 500))
        layer = features(x)

        # second rnn layer
        rnn = nn.RNN(input_size=self.feature_dim, hidden_size=500, num_layers=1, batch_first=True)
        h0 = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        output, hn = rnn(layer, h0)

        return output

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






