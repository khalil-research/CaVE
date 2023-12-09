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

from func import exactConeAlignedCosine, innerConeAlignedCosine
from pyepo.func import SPOPlus, perturbedFenchelYoung, NCE

def train(reg, optmodel, prob_name, mthd_name,
          loader_train, loader_train_ctr, loader_test, hparams):
    """
    A method to train and evaluate a neural net
    """
    # get training config
    config = hparams[prob_name][mthd_name]
    # start training
    tick = time.time()
    if mthd_name == "2s":
        # init loss
        mse = nn.MSELoss()
        # train
        train2S(reg, loader_train, mse, config.lr, config.epochs)
    elif mthd_name == "cave":
        # init loss
        cave = exactConeAlignedCosine(optmodel, solver=config.solver)
        # train
        trainCaVE(reg, loader_train_ctr, cave, config.lr, config.epochs)
    elif mthd_name == "cave+":
        # init loss
        cave = innerConeAlignedCosine(optmodel, solver=config.solver,
                                      max_iter=config.max_iter)
        # train
        trainCaVE(reg, loader_train_ctr, cave, config.lr, config.epochs)
    elif mthd_name == "caveh":
        # init loss
        cave = innerConeAlignedCosine(optmodel, solver=config.solver,
                                      solve_ratio=config.solve_ratio,
                                      inner_ratio=config.inner_ratio)
        # train
        trainCaVE(reg, loader_train_ctr, cave, config.lr, config.epochs)
    elif mthd_name == "spo+":
        # init loss
        spop = SPOPlus(optmodel)
        # train
        trainSPO(reg, loader_train, spop, config.lr, config.epochs)
    elif mthd_name == "pfyl":
        # init loss
        pfy = perturbedFenchelYoung(optmodel, n_samples=config.n_samples,
                                    sigma=config.sigma)
        # train
        trainPFYL(reg, loader_train, pfy, config.lr, config.epochs)
    elif mthd_name == "nce":
        nce = NCE(optmodel, solve_ratio=config.solve_ratio,
                  dataset=loader_train.dataset)
        # train
        trainNCE(reg, loader_train, nce, config.lr, config.epochs)
    else:
        message = "This algorithm {} is not yet implemented".format(mthd_name)
        raise NotImplementedError(message)
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
    # to DataFrame
    metrics = pd.DataFrame([metrics])
    # float
    metrics = metrics.astype(float)
    return metrics


def train2S(reg, dataloader, loss_func, lr, num_epochs):
    """
    A method for 2-stage training
    """
    print("2-Stage:")
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # train
    with tqdm(total=num_epochs*len(dataloader)) as tbar:
        for epoch in range(num_epochs):
            for data in dataloader:
                x, c, _, _ = data
                # predict
                cp = reg(x)
                # loss
                loss = loss_func(cp, c)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tbar.set_description("Epoch {:4.0f}, Loss: {:8.4f}".
                                     format(epoch, loss.item()))
                tbar.update(1)


def trainCaVE(reg, dataloader, loss_func, lr, num_epochs):
    """
    A method for CaVE training
    """
    print("CaVE:")
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # train
    with tqdm(total=num_epochs*len(dataloader)) as tbar:
        for epoch in range(num_epochs):
            for data in dataloader:
                x, _, _, ctr = data
                # predict
                cp = reg(x)
                # loss
                loss = loss_func(cp, ctr)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tbar.set_description("Epoch {:4.0f}, Loss: {:8.4f}".
                                     format(epoch, loss.item()))
                tbar.update(1)


def trainSPO(reg, dataloader, loss_func, lr, num_epochs):
    """
    A method for SPO+ training
    """
    print("SPO+:")
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # train
    with tqdm(total=num_epochs*len(dataloader)) as tbar:
        for epoch in range(num_epochs):
            for data in dataloader:
                x, c, w, z = data
                # predict
                cp = reg(x)
                # loss
                loss = loss_func(cp, c, w, z)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tbar.set_description("Epoch {:4.0f}, Loss: {:8.4f}".
                                     format(epoch, loss.item()))
                tbar.update(1)


def trainPFYL(reg, dataloader, loss_func, lr, num_epochs):
    """
    A method for PFYL training
    """
    print("PFYL:")
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # train
    with tqdm(total=num_epochs*len(dataloader)) as tbar:
        for epoch in range(num_epochs):
            for data in dataloader:
                x, _, w, _ = data
                # predict
                cp = reg(x)
                # loss
                loss = loss_func(cp, w)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tbar.set_description("Epoch {:4.0f}, Loss: {:8.4f}".
                                     format(epoch, loss.item()))
                tbar.update(1)


def trainNCE(reg, dataloader, loss_func, lr, num_epochs):
    """
    A method for NCE training
    """
    print("NCE:")
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # train
    with tqdm(total=num_epochs*len(dataloader)) as tbar:
        for epoch in range(num_epochs):
            for data in dataloader:
                x, _, w, _ = data
                # predict
                cp = reg(x)
                # loss
                loss = loss_func(cp, w)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tbar.set_description("Epoch {:4.0f}, Loss: {:8.4f}".
                                     format(epoch, loss.item()))
                tbar.update(1)
