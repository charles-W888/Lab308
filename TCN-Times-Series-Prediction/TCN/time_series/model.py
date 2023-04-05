# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/3/29 13:00
# @File     : model.py
# @Project  : TCN-Times-Series-Prediction
# @Software : PyCharm
# @Author   : Charles
import torch
from torch import nn
from TCN.tcn import TemporalConvNet


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1[:, :, -1])


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        """
        :param input_size: number of input features (1 for univariate forecasting)
        :param hidden_size: number of neurons in each hidden layer
        :param output_size: number of outputs to predict for each training example
        :param num_layers: number of lstm layers
        :param nhid: number of neurons in each hidden layer
        """
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        h = torch.randn(self.num_layers, batch_size, self.hidden_size).cuda()
        c = torch.randn(self.num_layers, batch_size, self.hidden_size).cuda()
        y1, _ = self.lstm(x, (h, c))
        y2 = self.linear(y1)
        return y2[:, -1, :]
