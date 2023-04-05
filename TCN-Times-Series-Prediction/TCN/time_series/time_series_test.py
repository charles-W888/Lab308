# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/3/29 12:59
# @File     : time_series_test.py
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

start_t = time.time()
parser = argparse.ArgumentParser(description='Long Term Forecast - Time Series Data')

# basic config
parser.add_argument('--seed', type=int, default=2000, help='random seed (default: 1111)')
parser.add_argument('--model', type=str, default='TCN', help='which model to use')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='report interval (default: 100')
parser.add_argument('--cuda', action='store_false', help='use CUDA (default: True)')

# data
parser.add_argument('--data_path', type=str, default='./data/testdata.csv', help='path of the data file')
parser.add_argument('--num_features', type=int, default=10, help='dimension of input sequence')
parser.add_argument('--scale', type=bool, default=True, help='use MinMaxScaler (default: True)')
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

# model define
parser.add_argument('--ksize', type=int, default=5, help='kernel size (default: 5)')  # 卷积核大小
parser.add_argument('--levels', type=int, default=6, help='# of levels (default: 8)')  # dilation层数
parser.add_argument('--nhid', type=int, default=80, help='number of hidden units per layer (default: 30, 80)')
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

# Note: We use a very simple setting here (assuming all levels have the same # of channels.
channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
dropout = args.dropout

model = TCN(args.seq_len, args.pred_len, channel_sizes, kernel_size=kernel_size, dropout=dropout)
if args.model == "TCN":
    model = model
elif args.model == "LSTM":
    model = LSTM(args.num_features, args.nhid, args.pred_len, args.n_layers)
else:
    print("-----Model's name is illegal!-----")

if args.cuda:
    model.cuda()
    X_train = X_train.cuda()
    Y_train = Y_train.cuda()
    X_vali = X_vali.cuda()
    Y_vali = Y_vali.cuda()
    X_test = X_test.cuda()
    Y_test = Y_test.cuda()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def train(epoch):
    global lr
    model.train()
    batch_idx = 1
    total_loss = 0
    for i in range(0, X_train.size(0), batch_size):
        if i + batch_size > X_train.size(0):
            x, y = X_train[i:], Y_train[i:]
        else:
            x, y = X_train[i:(i + batch_size)], Y_train[i:(i + batch_size)]
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            processed = min(i + batch_size, X_train.size(0))
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, X_train.size(0), 100. * processed / X_train.size(0), lr, cur_loss))
            total_loss = 0


def evaluate():
    model.eval()
    with torch.no_grad():
        output = model(X_test)
        test_loss = F.mse_loss(output[:, 0], Y_test[:, 0])
        test_val = Y_test[:, 0].cpu().detach().numpy()
        pred_val = output[:, 0].cpu().detach().numpy()
        print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.item()))
        return test_loss.item(), test_val, pred_val


for ep in range(1, epochs + 1):
    train(ep)

tloss, test_val, pred_val = evaluate()
scaled_true = scaler.inverse_transform(test_val.reshape(-1, 1)).reshape(-1)
scaled_pred = scaler.inverse_transform(pred_val.reshape(-1, 1)).reshape(-1)

filepath = './results/' + args.model + '_' + str(args.seq_len) + '_' + str(args.pred_len) + '.csv'
an = Analysis(scaled_true, scaled_pred, file_path=filepath)
an.save_result()
RMSE, MAE, R, R_2, sMAPE = an.metrics()
print('\nRMSE={}, MAE={}, R={}, R2={}, sMAPE={}'.format(RMSE, MAE, R, R_2, sMAPE))
an.fit_plot()

torch.cuda.empty_cache()
end_t = time.time()
print('[Cost time: {:.2f}(min)]'.format((end_t - start_t) / 60))
