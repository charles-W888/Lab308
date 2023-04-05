# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/3/29 13:00
# @File     : utils.py
# @Project  : TCN-Times-Series-Prediction
# @Software : PyCharm
# @Author   : Charles
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class Dataloader(Dataset):
    def __init__(self, size=None, data_path='./data/testdata.csv', scale=True):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        self.scale = scale
        self.data_path = data_path

    def read_data(self):
        self.scaler = MinMaxScaler()
        df_raw = pd.read_csv(self.data_path)
        col_name = df_raw.columns[1:]  # 去除时间列
        col_data = df_raw[col_name].values
        num_features = len(col_name)

        scaled_data = col_data
        if self.scale:
            for i in range(len(col_name)):
                scaled_data[:, i] = self.scaler.fit_transform(col_data[:, i].reshape(-1, 1)).reshape(-1)

        # 多步预测，X或Y的长度是data_len - input_seq_len - output_seq_len + 1
        L = len(df_raw) - self.seq_len - self.pred_len + 1
        data_X = np.zeros(shape=(L, self.seq_len, num_features))
        data_Y = np.zeros(shape=(L, self.pred_len))
        for i in range(L):
            data_X[i, :] = scaled_data[i: i + self.seq_len, :]
            data_Y[i] = scaled_data[i + self.seq_len: i + self.seq_len + self.pred_len, -1]

        ptr = int(0.7 * len(df_raw))
        ptr1 = int(0.8 * len(df_raw))
        train_X = data_X[:ptr]
        train_Y = data_Y[:ptr]
        vali_X = data_X[ptr:ptr1]
        vali_Y = data_Y[ptr:ptr1]
        test_X = data_X[ptr1:]
        test_Y = data_Y[ptr1:]

        train_X = torch.Tensor(train_X)
        train_Y = torch.Tensor(train_Y)
        vali_X = torch.Tensor(vali_X)
        vali_Y = torch.Tensor(vali_Y)
        test_X = torch.Tensor(test_X)
        test_Y = torch.Tensor(test_Y)
        return train_X, train_Y, vali_X, vali_Y, test_X, test_Y, self.scaler


class Analysis:
    def __init__(self, true, pred, file_path='./results/TCN_96_24.csv'):
        self.true = true
        self.pred = pred
        self.file_path = file_path

    def fit_plot(self):
        plt.figure()
        plt.plot(self.pred[-2000:-1], label='Prediction', linewidth=1)
        plt.plot(self.true[-2000:-1], label='Observed', linewidth=1)
        plt.title('The curve fitting of data')
        plt.legend()
        plt.savefig('prediction_TCN.pdf', bbox_inches='tight')
        # plt.show()

    def metrics(self):
        tx = self.true - np.mean(self.true)
        ty = self.pred - np.mean(self.pred)
        RMSE = np.sqrt(np.mean(np.square(self.pred - self.true)))
        MAE = np.mean(np.abs(self.pred - self.true))
        R = np.sum(tx * ty) / (np.sqrt(np.sum(tx ** 2)) * np.sqrt(np.sum(ty ** 2)))
        R_2 = 1 - np.sum((self.true - self.pred) ** 2) / np.sum((self.pred - np.mean(self.true)) ** 2)
        sMAPE = np.mean(np.abs((self.true - self.pred) / ((np.abs(self.true) + np.abs(self.pred)) / 2)))
        return RMSE, MAE, R, R_2, sMAPE

    def save_result(self):
        f = pd.DataFrame({'True': self.true, 'Pred': self.pred})
        f.to_csv(self.file_path, index=False)
