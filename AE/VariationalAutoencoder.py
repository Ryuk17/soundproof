"""
@ Filename:       VariationalAutoencoder.py
@ Author:         Danc1elion
@ Create Date:    2020-01-08   
@ Update Date:    2020-01-08 
@ Description:    Implement VariationalAutoencoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self, in_size, out_size):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(in_size, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, out_size)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)


    def reparameterize(self, mu, var):
        std = var.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decoder(self, x):
        h = F.relu(self.fc3(x))
        return F.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, var = self.encoder(x)
        y = self.reparameterize(mu, var)
        return self.decoder(y), mu, var


def lossFunction(gen, org, mu, var):
    """
    loss function for VAE
    :param gen: generated data
    :param org: org data
    :param mu: latent mean
    :param var: latent var
    :return: loss
    """

    loss1 = nn.BCELoss(gen, org)
    KLD = mu.pow(2).add_(var.exp()).mul_(-1).add_(1).add_(var)
    loss2 = torch.sum(KLD).mul_(-0.5)
    return loss1 + loss2


def main():
    x = torch.randn(10, 784)
    net = VAE(784, 784)
    print(net(x))


if __name__ == '__main__':
    main()