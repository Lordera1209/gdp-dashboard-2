#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Version  : 3.11
# @Author   : Lordera
# @Datetime : 2024/5/16 下午1:54
# @Project  : multi-classify
# @File     : model_attention.py


import torch
import torch.nn as nn
import math


# 提供掩模功能(sequence_mask & masked_softmax)
def sequence_mask(X, valid_len, value):
    # valid_len一定是1维
    # X是2维时，valid_len为(1 * batch_size); X是3维时，valid_len为(batch_size * num_steps)
    max_len = X.size(1)
    mask = torch.arange(max_len, dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_len, value):
    # 接收3维，变为2维处理
    if valid_len is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_len.dim() == 1:
            valid_len = torch.repeat_interleave(valid_len, shape[1])
        else:
            valid_len = valid_len.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_len, value)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


# 单头注意力
class DotProductAttention(nn.Module):
    def __init__(self, config):
        super(DotProductAttention, self).__init__()
        self.attention_weights = None
        self.dropout = nn.Dropout(config.dropout)
        self.masked_value = config.masked_value
        self.device = config.device

    # 'query': (batch_size, queries, hidden_size)
    # 'key': (batch_size, kv_pairs, hidden_size)
    # 'value': (batch_size, kv_pairs, hidden_size)

    def forward(self, query, key, value, valid_len):
        d = query.shape[-1]
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = self.dropout(masked_softmax(scores, valid_len, self.masked_value))
        return torch.bmm(self.attention_weights, value)


# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, config, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.device = config.device
        self.num_heads = config.num_heads
        self.attention = DotProductAttention(config=config)
        self.W_q = nn.Linear(config.hidden_size, config.hidden_size, bias=bias).to(config.device)
        self.W_k = nn.Linear(config.hidden_size, config.hidden_size, bias=bias).to(config.device)
        self.W_v = nn.Linear(config.hidden_size, config.hidden_size, bias=bias).to(config.device)
        self.W_o = nn.Linear(config.hidden_size, config.hidden_size, bias=bias).to(config.device)

    def forward(self, query, key, value, valid_len):
        Q = transpose_qkv(self.W_q(query), self.num_heads)
        K = transpose_qkv(self.W_k(key), self.num_heads)
        V = transpose_qkv(self.W_v(value), self.num_heads)

        if valid_len is not None:
            if valid_len.ndim == 1:
                valid_len = valid_len.repeat(self.num_heads)
            else:
                valid_len = valid_len.repeat(self.num_heads, 1)

        # 'output' shape: (batch_size * num_heads, max_len, hidden_size / num_heads)
        output = self.attention(Q, K, V, valid_len)

        # 'output_concat' shape: (batch_size, max_len, hidden_size)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


# 多头注意力中模型矩阵变换 & 模型矩阵还原
def transpose_qkv(X, num_heads):
    # Input 'X' shape: (batch_size, max_len, hidden_size).
    # Output 'X' shape:
    # (batch_size, max_len, num_heads, hidden_size / num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 'X' shape:
    # (batch_size, num_heads, max_len, hidden_size / num_heads)
    X = X.permute(0, 2, 1, 3)

    # 'output' shape:
    # (batch_size * num_heads, max_len, hidden_size / num_heads)
    output = X.reshape(-1, X.shape[2], X.shape[3])
    return output


def transpose_output(X, num_heads):
    # A reversed version of `transpose_qkv`
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
