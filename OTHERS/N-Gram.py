"""
@FileName: N-Gram.py
@Description: Implement N-Gram
@Author : Lj
@CreateDate: 2019/11/15 16:03
@LastEditTime: 2019/11/15 16:03
@LastEditors: Please set LastEditors
@Version: v1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Ngram(nn.Module):
    def __init__(self, vocb_size, context_size, n_dim):
        super(Ngram, self).__init__()

        self.n_word = vocb_size
        self.embedding = nn.Embedding(self.n_word, n_dim)
        self.layer1 = nn.Linear(context_size * n_dim, 128)
        self.layer2 = nn.Linear(128, self.n_word)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(emb.size(0), -1)
        out = self.layer1(emb)
        out = self.layer2(out)
        res = F.log_softmax(out)
        return res

