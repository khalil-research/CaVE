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

from src.cave import exactConeAlignedCosine, innerConeAlignedCosine
from pyepo.func import SPOPlus, perturbedFenchelYoung, NCE
from pyepo.model.grb import tspMTZModel
import metric

def train(reg, optmodel, prob_name, mthd_name,
          loader_train, loader_train_ctr, loader_test, hparams, relaxed):
    """
    A method to train and evaluate a neural net
    """
    if prob_name[:3] != "tsp" and relaxed:
        raise Exception("Relaxed model does not exist")
    elif prob_name[:3] == "tsp" and relaxed:
        print("Using relaxation of TSP-MTZ for training...")
        optmodel_rel = tspMTZModel(optmodel.num_nodes).relax()
    # get training config
    config = hparams[prob_name][mthd_name]
    # start training
    tick = time.time()
    if mthd_name == "2s":
        # init loss
        mse = nn.MSELoss()
        # train
        loss_log = train2S(reg, loader_train, mse, config.lr, config.epochs)
    elif mthd_name == "cave":
        # init loss
        cave = exactConeAlignedCosine(optmodel, solver=config.solver)
        # train
        loss_log = trainCaVE(reg, loader_train_ctr, cave, config.lr, config.epochs)
    elif mthd_name == "cave+":
        # init loss
        cave = innerConeAlignedCosine(optmodel, solver=config.solver,
                                      max_iter=config.max_iter)
        # train
        loss_log = trainCaVE(reg, loader_train_ctr, cave, config.lr, config.epochs)
    elif mthd_name == "caveh":
        # init loss
        cave = innerConeAlignedCosine(optmodel, solver=config.solver,
                                      solve_ratio=config.solve_ratio,
                                      inner_ratio=config.inner_ratio)
        # train
        loss_log = trainCaVE(reg, loader_train_ctr, cave, config.lr, config.epochs)
    elif mthd_name == "spo+":
        # init loss
        if relaxed:
            spop = SPOPlus(optmodel_rel)
        else:
            spop = SPOPlus(optmodel)
        # train
        loss_log = trainSPO(reg, loader_train, spop, config.lr, config.epochs)
    elif mthd_name == "pfyl":
        # init loss
        if relaxed:
            pfy = perturbedFenchelYoung(optmodel_rel, n_samples=config.n_samples,
                                        sigma=config.sigma)
        else:
            pfy = perturbedFenchelYoung(optmodel, n_samples=config.n_samples,
                                        sigma=config.sigma)
        # train
        loss_log = trainPFYL(reg, loader_train, pfy, config.lr, config.epochs)
    elif mthd_name == "nce":
        nce = NCE(optmodel, solve_ratio=config.solve_ratio,
                  dataset=loader_train.dataset)
        # train
        loss_log = trainNCE(reg, loader_train, nce, config.lr, config.epochs)
    else:
        message = "This algorithm {} is not yet implemented".format(mthd_name)
        raise NotImplementedError(message)
    tock = time.time()
    # record time
    elapsed_train = tock - tick
    # regret
    print("Evaluate training set...")
    regret_train, nodes_train, _ = metric.regret(reg, optmodel, loader_train)
    print("Evaluate test set...")
    tick = time.time()
    regret_test, nodes_test, instance_res = metric.regret(reg, optmodel, loader_test)
    tock = time.time()
    elapsed_test = tock - tick
    # mse
    mse_train = pyepo.metric.MSE(reg, loader_train)
    mse_test = pyepo.metric.MSE(reg, loader_test)
    # output
    metrics = {"Train Regret":None, "Test Regret":None,
               "Train MSE":None, "Test MSE":None,
               "Train Elapsed":None, "Test Elapsed":None,
               "Train Nodes Count":None, "Test Nodes Count":None}
    metrics["Train Elapsed"] = elapsed_train
    metrics["Train Regret"] = regret_train
    metrics["Train MSE"] = mse_train
    metrics["Train Nodes Count"] = nodes_train
    metrics["Test Elapsed"] = elapsed_test
    metrics["Test Regret"] = regret_test
    metrics["Test MSE"] = mse_test
    metrics["Test Nodes Count"] = nodes_test
    # to DataFrame
    metrics = pd.DataFrame([metrics])
    return metrics, loss_log, instance_res


def train2S(reg, dataloader, loss_func, lr, num_epochs):
    """
    A method for 2-stage training
    """
    print("2-Stage:")
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # init log
    loss_log = []
    # train
    with tqdm(total=num_epochs*len(dataloader)) as tbar:
        for epoch in range(num_epochs):
            for data in dataloader:
                x, c, _, _ = data
                # predict
                cp = reg(x)
                # loss
                loss = loss_func(cp, c)
                loss_log.append(loss.item())
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tbar.set_description("Epoch {:4.0f}, Loss: {:8.4f}".
                                     format(epoch, loss.item()))
                tbar.update(1)
    return loss_log


def trainCaVE(reg, dataloader, loss_func, lr, num_epochs):
    """
    A method for CaVE training
    """
    print("CaVE:")
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # init log
    loss_log = []
    # train
    with tqdm(total=num_epochs*len(dataloader)) as tbar:
        for epoch in range(num_epochs):
            for data in dataloader:
                x, _, _, ctr = data
                # predict
                cp = reg(x)
                # loss
                loss = loss_func(cp, ctr)
                loss_log.append(loss.item())
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tbar.set_description("Epoch {:4.0f}, Loss: {:8.4f}".
                                     format(epoch, loss.item()))
                tbar.update(1)
    return loss_log


def trainSPO(reg, dataloader, loss_func, lr, num_epochs):
    """
    A method for SPO+ training
    """
    print("SPO+:")
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # init log
    loss_log = []
    # train
    with tqdm(total=num_epochs*len(dataloader)) as tbar:
        for epoch in range(num_epochs):
            for data in dataloader:
                x, c, w, z = data
                # predict
                cp = reg(x)
                # loss
                loss = loss_func(cp, c, w, z)
                loss_log.append(loss.item())
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tbar.set_description("Epoch {:4.0f}, Loss: {:8.4f}".
                                     format(epoch, loss.item()))
                tbar.update(1)
    return loss_log


def trainPFYL(reg, dataloader, loss_func, lr, num_epochs):
    """
    A method for PFYL training
    """
    print("PFYL:")
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # init log
    loss_log = []
    # train
    with tqdm(total=num_epochs*len(dataloader)) as tbar:
        for epoch in range(num_epochs):
            for data in dataloader:
                x, _, w, _ = data
                # predict
                cp = reg(x)
                # loss
                loss = loss_func(cp, w)
                loss_log.append(loss.item())
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tbar.set_description("Epoch {:4.0f}, Loss: {:8.4f}".
                                     format(epoch, loss.item()))
                tbar.update(1)
    return loss_log


def trainNCE(reg, dataloader, loss_func, lr, num_epochs):
    """
    A method for NCE training
    """
    print("NCE:")
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # init log
    loss_log = []
    # train
    with tqdm(total=num_epochs*len(dataloader)) as tbar:
        for epoch in range(num_epochs):
            for data in dataloader:
                x, _, w, _ = data
                # predict
                cp = reg(x)
                # loss
                loss = loss_func(cp, w)
                loss_log.append(loss.item())
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tbar.set_description("Epoch {:4.0f}, Loss: {:8.4f}".
                                     format(epoch, loss.item()))
                tbar.update(1)
    return loss_log
