"""
@FileName: VAE.py
@Description: Implement VAE
@Author : Ryuk
@CreateDate: 2019/11/15 16:28
@LastEditTime: 2019/11/15 16:28
@LastEditors: Please set LastEditors
@Version: v1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class VariationalAutoEncoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(VariationalAutoEncoder, self).__init__()

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
    net = VariationalAutoEncoder(784, 784)
    print(net(x))


if __name__ == '__main__':
    main()