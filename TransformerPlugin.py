#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : ConvTransformerTS
@ FileName: transformer_singlestep.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 1/3/22 11:12
"""

import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from helper import series_to_supervised, stage_series_to_supervised
import PyPluMA

torch.manual_seed(0)
np.random.seed(0)

# This concept is also called teacher forcing.
# The flag decides if the loss will be calculated over all or just the predicted values.
calculate_loss_over_all_values = False
criterion = nn.MSELoss()
# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

# src = torch.rand((10, 32, 512)) # (S,N,E)
# tgt = torch.rand((20, 32, 512)) # (T,N,E)
# out = transformer_model(src, tgt)
# print(out)

input_window = 84
output_window = 12
batch_size = 512       # batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size=8, num_layers=2, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

model = TransAm().to(device)

# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = np.append(input_data[i:i + tw][:-output_window], output_window * [0])
        train_label = input_data[i:i+tw]
        # train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)


def get_data(inputfile, divide, column,pct):
    # time = np.arange(0, 400, 0.1)
    # amplitude = np.sin(time) + np.sin(time * 0.05) + np.sin(time * 0.12) * np.random.normal(-0.2, 0.2, len(time))
    dataset = pd.read_csv(inputfile, index_col=0)  # header=0, parse_dates=True, squeeze=True
    dataset.fillna(0, inplace=True)
    data = dataset.iloc[:divide, column]  # 'WS_S1' - 0th column
    values = data.to_numpy()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    # amplitude = self.scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)
    amplitude = scaler.fit_transform(values.reshape(-1, 1))

    sampels = int(len(dataset)*pct)
    train_data = amplitude[:sampels]
    test_data = amplitude[sampels:]

    # convert our train data into a pytorch train tensor
    # train_tensor = torch.FloatTensor(train_data).view(-1)
    train_sequence = create_inout_sequences(train_data, input_window)
    train_sequence = train_sequence[:-output_window]

    # test_data = torch.FloatTensor(test_data).view(-1)
    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]

    return train_sequence.to(device), test_data.to(device), scaler


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target


def train(train_data, epoch):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)

        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ''lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(epoch, batch, len(train_data) // batch_size,
                scheduler.get_lr()[0], elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def plot_and_loss(eval_model, data_source, epoch, scaler):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            output = eval_model(data)
            if calculate_loss_over_all_values:
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()

            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    # test_result = test_result.cpu().numpy()
    # len(test_result)
    test_result = test_result.reshape(-1, 1)
    truth = truth.reshape(-1, 1)
    inv_yhat = scaler.inverse_transform(test_result)   # test_result[:800]
    inv_y = scaler.inverse_transform(truth)
    print(test_result.shape, truth.shape)

    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    mae = mean_absolute_error(inv_y, inv_yhat)
    # mape = mean_absolute_percentage_error(truth, test_result)
    print('Test RMSE: %.3f' % rmse)
    print('Test MAE: %.3f' % mae)
    # print('Test MAPE: %.3f' % mape)

    # pyplot.rcParams['font.family'] = 'serif'
    # pyplot.rcParams['font.serif'] = ['Times New Roman'] + pyplot.rcParams['font.serif']

    #date = ['09/10', '09/11', '09/12', '09/13', '09/14', '09/15', '09/16']
    #pyplot.rcParams["figure.figsize"] = (8, 6)
    #pyplot.plot(inv_yhat[99635:100499], label='prediction')
    #pyplot.plot(inv_y[99635:100499], label='truth')
    #pyplot.title("Predicted & Actual Value of WS_S1", fontsize='18')
    #pyplot.xlabel('Time', fontsize='16')
    #pyplot.ylabel('Water Stage', fontsize='16')
    # pyplot.xticks(np.arange(99635, 100500, 144), date, fontsize=14)
    #pyplot.legend(prop={"size": 14})
    #pyplot.xticks(fontsize=14)
    #pyplot.yticks(fontsize=14)
    # pyplot.savefig('graph/lstm_prediction.png', dpi=300)

    # pyplot.plot(test_result - truth, color="green")
    # pyplot.grid(True, which='both')
    # pyplot.axhline(y=0, color='k')
    # pyplot.savefig('graph/transformer-epoch%d.png' % epoch)
    #pyplot.show()
    #pyplot.close()

    return total_loss / i


def predict_future(eval_model, data_source, steps):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    _, data = get_batch(data_source, 0, 1)
    with torch.no_grad():
        for i in range(0, steps, 1):
            input = torch.clone(data[-input_window:])
            input[-output_window:] = 0
            output = eval_model(data[-input_window:])
            data = torch.cat((data, output[-1:]))

    data = data.cpu().view(-1)

    pyplot.plot(data, color="red")
    pyplot.plot(data[:input_window], color="blue")
    # pyplot.grid(True, which='both')
    # pyplot.axhline(y=0, color='k')
    # pyplot.savefig('graph/transformer-future%d.png' % steps)
    pyplot.show()
    pyplot.close()


def evaluate(eval_model, data_source):
    eval_model.eval()       # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
            if calculate_loss_over_all_values:
                total_loss += len(data[0]) * criterion(output, targets).cpu().item()
            else:
                total_loss += len(data[0]) * criterion(output[-output_window:], targets[-output_window:]).cpu().item()
    return total_loss / len(data_source)


class TransformerPlugin:
    def input(self, filename):
        infile = open(filename, 'r')
        self.parameters = dict()
        for line in infile:
            contents = line.strip().split('\t')
            self.parameters[contents[0]] = contents[1]

        self.train_data, self.val_data, self.scaler = get_data(PyPluMA.prefix()+"/"+self.parameters["inputfile"], int(self.parameters["divide"]), int(self.parameters["column"]), float(self.parameters["pct"]))

    def run(self):

        lr = float(self.parameters["lr"])
        global optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        global scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

        best_val_loss = float("inf")
        epochs = int(self.parameters["epochs"])  # The number of epochs
        best_model = None

        self.all_train_loss, self.all_val_loss = [], []

        for epoch in range(1, epochs + 1):
           epoch_start_time = time.time()
           train(self.train_data, epoch)

           if epoch % 30 == 0:
             val_loss = plot_and_loss(model, self.val_data, epoch, self.scaler)
             # predict_future(model, self.val_data, 200)
           else:
             train_loss = evaluate(model, self.train_data)
             self.all_train_loss.append(train_loss)

             val_loss = evaluate(model, self.val_data)
             self.all_val_loss.append(val_loss)

             print('-' * (epochs-1))
             print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), train_loss, math.exp(train_loss)))
             print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
             print('-' * (epochs-1))

             # if val_loss < best_val_loss:
             #    best_val_loss = val_loss
             #    best_model = model

             scheduler.step()

    def output(self, filename):
      pyplot.plot(self.all_train_loss, label='train')
      pyplot.plot(self.all_val_loss, label='test')
      outf = open(filename+".txt", 'w')
      outf.write(str(self.all_val_loss))
      pyplot.xlabel('Loss', fontsize=16)
      pyplot.ylabel('Epoch', fontsize=16)
      pyplot.legend()
      pyplot.savefig(filename)
      pyplot.show()

# src = torch.rand(input_window, batch_size, 1) # (source sequence length,batch size,feature number)
# out = model(src)
#
# print(out)
# print(out.shape)
