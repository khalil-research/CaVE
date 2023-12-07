#!/usr/bin/env python
# coding: utf-8
"""
Training
"""

import time
from tqdm import tqdm

import pandas as pd
import pyepo
import torch
from torch import nn

from config import configs

def train(reg, optmodel, prob_name, mthd_name,
          loader_train, loader_train_ctr, loader_test):
    """
    A method to train and evaluate a neural net
    """
    # get training config
    config = configs[prob_name][mthd_name]
    # start training
    tick = time.time()
    if mthd_name == "2s":
        train2S(reg, loader_train_ctr, config.lr, config.epochs)
    tock = time.time()
    # record time
    elapsed_train = tock - tick
    # regret
    regret_train = pyepo.metric.regret(reg, optmodel, loader_train)
    tick = time.time()
    regret_test = pyepo.metric.regret(reg, optmodel, loader_test)
    tock = time.time()
    elapsed_test = tock - tick
    # mse
    mse_train = pyepo.metric.MSE(reg, loader_train)
    mse_test = pyepo.metric.MSE(reg, loader_test)
    # output
    metrics = {"Train Regret":None, "Test Regret":None,
           "Train MSE":None, "Test MSE":None,
           "Train Elapsed":None, "Test Elapsed":None}
    metrics["Train Elapsed"] = elapsed_train
    metrics["Train Regret"] = regret_train
    metrics["Train MSE"] = mse_train
    metrics["Test Elapsed"] = elapsed_test
    metrics["Test Regret"] = regret_test
    metrics["Test MSE"] = mse_test
    metrics = pd.DataFrame([metrics])
    return metrics


def train2S(reg, dataloader, lr, num_epochs):
    """
    A method for 2-stage training
    """
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # init loss
    mse = nn.MSELoss()
    # train
    with tqdm(total=num_epochs*len(dataloader)) as tbar:
        for epoch in range(num_epochs):
            for data in dataloader:
                x, c, _, _ = data
                # predict
                cp = reg(x)
                # loss
                loss = mse(cp, c)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tbar.set_description("Epoch {:4.0f}, Loss: {:8.4f}".
                                     format(epoch, loss.item()))
                tbar.update(1)
