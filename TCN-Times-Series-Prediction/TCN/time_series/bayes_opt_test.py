# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/3/31 18:55
# @File     : bayes_opt_test.py
# @Project  : TCN-Times-Series-Prediction
# @Software : PyCharm
# @Author   : Charles
import argparse
import time
import sys

sys.path.append("../../")

import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from TCN.time_series.model import TCN, LSTM
from TCN.time_series.utils import Dataloader, Analysis
from bayes_opt import BayesianOptimization

start_t = time.time()
parser = argparse.ArgumentParser(description='Long Term Forecast - Time Series Data')

# basic config
parser.add_argument('--seed', type=int, default=2000, help='random seed (default: 1111)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='report interval (default: 100')
parser.add_argument('--cuda', action='store_false', help='use CUDA (default: True)')

# data
parser.add_argument('--data_path', type=str, default='./data/testdata.csv', help='path of the data file')
parser.add_argument('--num_features', type=int, default=10, help='dimension of input sequence')
parser.add_argument('--scale', type=bool, default=True, help='use MinMaxScaler (default: True)')
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

# model define
parser.add_argument('--ksize', type=int, default=6, help='kernel size (default: 5)')
parser.add_argument('--levels', type=int, default=6, help='# of levels (default: 8)')  # dilation层数
parser.add_argument('--nhid', type=int, default=30, help='number of hidden units per layer (default: 30, 80)')
parser.add_argument('--n_layers', type=int, default=3, help='number of lstm layers (default: 2)')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout applied to layers (default: 0.1)')
parser.add_argument('--clip', type=float, default=-1, help='gradient clip, -1 means no clip (default: -1)')

# optimization
parser.add_argument('--epochs', type=int, default=10, help='upper epoch limit (default: 10)')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size (default: 32)')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate (default: 1e-4)')
parser.add_argument('--optim', type=str, default='Adam', help='optimizer to use (default: Adam)')

args = parser.parse_args()

random.seed(80)
np.random.seed(80)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

batch_size = args.batch_size
epochs = args.epochs

print("Args in experiment:")
print(args)
size = [args.seq_len, args.pred_len]
data_loader = Dataloader(size=size, data_path=args.data_path, scale=args.scale)
X_train, Y_train, X_vali, Y_vali, X_test, Y_test, scaler = data_loader.read_data()

if args.cuda:
    X_train = X_train.cuda()
    Y_train = Y_train.cuda()
    X_vali = X_vali.cuda()
    Y_vali = Y_vali.cuda()
    X_test = X_test.cuda()
    Y_test = Y_test.cuda()

lr = args.lr


def bayes_optimize_model(kernel_size, levels, nhid, dropout):
    # LSTM优化调参时，把函数参数改为：(nhid, n_layers)即可
    global lr
    kernel_size = round(kernel_size)  # 四舍五入取整
    levels = round(levels)
    nhid = round(nhid)
    channel_sizes = [nhid] * levels
    dropout = round(dropout, 2)
    # n_layers = round(n_layers)

    model1 = TCN(args.seq_len, args.pred_len, channel_sizes, kernel_size=kernel_size, dropout=dropout)
    # model1 = LSTM(args.num_features, nhid, args.pred_len, n_layers)
    model1.cuda()
    optimizer = getattr(optim, args.optim)(model1.parameters(), lr=lr)
    model1.train()
    batch_idx = 1
    total_loss = 0
    for i in range(0, X_train.size(0), batch_size):
        if i + batch_size > X_train.size(0):
            x, y = X_train[i:], Y_train[i:]
        else:
            x, y = X_train[i:(i + batch_size)], Y_train[i:(i + batch_size)]
        optimizer.zero_grad()
        output = model1(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model1.parameters(), args.clip)
        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            processed = min(i + batch_size, X_train.size(0))
            # print('Train: [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
            #     processed, X_train.size(0), 100. * processed / X_train.size(0), lr, cur_loss))
            total_loss = 0
    model1.eval()
    with torch.no_grad():
        output = model1(X_vali)
        val_loss = F.mse_loss(output[:, 0], Y_vali[:, 0])
        return -val_loss.item()  # 因为BayesianOptimization以最大值为目标


if __name__ == "__main__":
    model_bo = BayesianOptimization(bayes_optimize_model,
                                    pbounds={'kernel_size': (3, 9),
                                             'levels': (3, 9),
                                             'nhid': (2, 100),
                                             'dropout': (0.0, 0.3)})
                                             # 'n_layers': (1, 20)})
    model_bo.maximize(n_iter=100, init_points=10)
    # print(model_bo.max)
    with open('bayes_info.txt', 'w+') as f:
        title = "| {: ^12} | {: ^12} | {: ^12} | {: ^12} | {: ^12} |".format(
            "target", "dropout", "kernel...", "levels", "nhid")
        split = "{:-^66}".format("")
        f.write(title)
        f.write('\n')
        f.write(split)
        f.write('\n')
        value = "| {: ^12} | {: ^12} | {: ^12} | {: ^12} | {: ^12} |".format(
            round(model_bo.max['target'], 5), round(model_bo.max['params']['dropout'], 2), round(model_bo.max['params']['kernel_size']),
            round(model_bo.max['params']['levels']), round(model_bo.max['params']['nhid']))
        f.write(value)
        f.write('\n')

    torch.cuda.empty_cache()
    end_t = time.time()
    print('[Cost time: {:.2f}(min)]'.format((end_t - start_t) / 60))
