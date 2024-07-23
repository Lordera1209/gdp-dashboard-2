#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Version  : 3.11
# @Author   : Lordera
# @Datetime : 2024/5/20 下午5:37
# @Project  : multi-reference
# @File     : FastText.py

import torch.nn as nn
import torch.nn.functional as F
from Attention import MultiHeadAttention


class Model(nn.Module):
    def __init__(self, config, index):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(
            config.n_vocab,
            config.embed,
            padding_idx=config.n_vocab - 1
        ).to(config.device)
        self.dropout = nn.Dropout(config.dropout).to(config.device)
        self.fc1 = nn.Linear(config.embed, config.hidden_size).to(config.device)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes[index]).to(config.device)


    def forward(self, x):
        out = self.embedding(x[0])
        out = self.dropout(self.fc1(out))
        out = F.relu(out)
        out = out.mean(dim=1)
        out = self.fc2(out)
        return out


class Attention_Model(nn.Module):
    def __init__(self, config):
        super(Attention_Model, self).__init__()
        self.embedding = nn.Embedding(
            config.n_vocab,
            config.embed,
            padding_idx=config.n_vocab - 1
        ).to(config.device)
        self.dropout = nn.Dropout(config.dropout).to(config.device)
        self.fc1 = nn.Linear(config.embed, config.hidden_size).to(config.device)
        self.fc2 = nn.Linear(2 * config.hidden_size, config.hidden_size).to(config.device)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, config.num_layers, bidirectional=True,
                            batch_first=True, dropout=config.dropout).to(config.device)
        self.attention = MultiHeadAttention(config).to(config.device)
        self.dropout = nn.Dropout(config.dropout).to(config.device)
        self.fc3 = nn.Linear(config.hidden_size, config.num_classes[10]).to(config.device)

    def forward(self, x):
        out = self.embedding(x[0])
        out = self.dropout(self.fc1(out))
        out, _ = self.lstm(out)
        out = self.fc2(out)
        out = self.attention(out, out, out, x[1])
        out = F.relu(out)
        out = out.mean(dim=1)
        out = self.fc3(out)
        return out
