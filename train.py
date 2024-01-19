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
from gurobipy import GRB

from src.cave import exactConeAlignedCosine, innerConeAlignedCosine
from pyepo.func import SPOPlus, perturbedFenchelYoung, NCE
from pyepo.model.grb import tspMTZModel
from src.model import vrpModel2
import metric

def train(reg, optmodel, prob_name, mthd_name, loader_train, loader_val, loader_test,
          hparams, relaxed, rlog):
    """
    A method to train and evaluate a neural net
    """
    if prob_name[:3] == "tsp" and relaxed:
        print("Using relaxation of TSP-MTZ for training...")
        optmodel_rel = tspMTZModel(optmodel.num_nodes).relax()
    elif prob_name[:3] == "vrp" and relaxed:
        print("Using relaxation of vrp for training...")
        optmodel_rel = vrpModel2(optmodel.num_nodes, optmodel.demands,
                                 optmodel.capacity, optmodel.num_vehicle).relax()
    elif relaxed:
        raise Exception("Relaxed model does not exist")
    # get training config
    config = hparams[prob_name][mthd_name]
    # set timelimit
    if prob_name == "vrp20":
        optmodel._model.Params.timelimit = 30
    # start training
    tick = time.time()
    if mthd_name == "2s":
        # init loss
        mse = nn.MSELoss()
        # train
        loss_log, regret_log = train2S(optmodel, reg, loader_train, loader_val,
                                       mse, config.lr, config.epochs, rlog)
    elif mthd_name == "cave":
        # init loss
        cave = exactConeAlignedCosine(optmodel, solver=config.solver)
        # train
        loss_log, regret_log = trainCaVE(optmodel, reg, loader_train, loader_val,
                                         cave, config.lr, config.epochs, rlog)
    elif mthd_name == "cave+":
        # init loss
        cave = innerConeAlignedCosine(optmodel, solver=config.solver)
        # train
        loss_log, regret_log = trainCaVE(optmodel, reg, loader_train, loader_val,
                                         cave, config.lr, config.epochs, rlog)
    elif mthd_name == "caveh":
        # init loss
        cave = innerConeAlignedCosine(optmodel, solver=config.solver,
                                      solve_ratio=config.solve_ratio,
                                      inner_ratio=config.inner_ratio)
        # train
        loss_log, regret_log = trainCaVE(optmodel, reg, loader_train, loader_val,
                                         cave, config.lr, config.epochs, rlog)
    elif mthd_name == "spo+":
        # init loss
        if relaxed:
            spop = SPOPlus(optmodel_rel)
        else:
            spop = SPOPlus(optmodel)
        # train
        loss_log, regret_log = trainSPO(optmodel, reg, loader_train, loader_val,
                                        spop, config.lr, config.epochs, rlog)
    elif mthd_name == "pfyl":
        # init loss
        if relaxed:
            pfy = perturbedFenchelYoung(optmodel_rel, n_samples=config.n_samples,
                                        sigma=config.sigma)
        else:
            pfy = perturbedFenchelYoung(optmodel, n_samples=config.n_samples,
                                        sigma=config.sigma)
        # train
        loss_log, regret_log = trainPFYL(optmodel, reg, loader_train, loader_val,
                                         pfy, config.lr, config.epochs, rlog)
    elif mthd_name == "nce":
        nce = NCE(optmodel, solve_ratio=config.solve_ratio,
                  dataset=loader_train.dataset)
        # train
        loss_log, regret_log = trainNCE(optmodel, reg, loader_train, loader_val,
                                        nce, config.lr, config.epochs, rlog)
    else:
        message = "This algorithm {} is not yet implemented".format(mthd_name)
        raise NotImplementedError(message)
    tock = time.time()
    # remove timelimit
    if prob_name == "vrp20":
        optmodel._model.Params.timelimit = GRB.INFINITY
    # record time
    elapsed_train = tock - tick
    # regret
    print("Evaluate training set...")
    regret_train, mse_train, nodes_train, _ = metric.regret(reg, optmodel, loader_train)
    print("Evaluate test set...")
    tick = time.time()
    regret_test, mse_test, nodes_test, instance_res = metric.regret(reg, optmodel, loader_test)
    tock = time.time()
    elapsed_test = tock - tick
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
    return metrics, loss_log, regret_log, instance_res


def train2S(optmodel, reg, dataloader, dataloader_val, loss_func,
            lr, num_epochs, rlog=False):
    """
    A method for 2-stage training
    """
    print("2-Stage:")
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # init log
    loss_log = []
    if rlog:
        regret, _, _, _ = metric.regret(reg, optmodel, dataloader_val)
        regret_log = [regret]
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
            if rlog:
                regret, _, _, _ = metric.regret(reg, optmodel, dataloader_val)
                regret_log.append(regret)
    return loss_log, regret_log


def trainCaVE(optmodel, reg, dataloader, dataloader_val, loss_func,
              lr, num_epochs, rlog=False):
    """
    A method for CaVE training
    """
    print("CaVE:")
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # init log
    loss_log = []
    if rlog:
        regret, _, _, _ = metric.regret(reg, optmodel, dataloader_val)
        regret_log = [regret]
    # train
    with tqdm(total=num_epochs*len(dataloader)) as tbar:
        for epoch in range(num_epochs):
            for data in dataloader:
                x, _, _, _, ctr = data
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
            if rlog:
                regret, _, _, _ = metric.regret(reg, optmodel, dataloader_val)
                regret_log.append(regret)
    return loss_log, regret_log


def trainSPO(optmodel, reg, dataloader, dataloader_val, loss_func,
             lr, num_epochs, rlog=False):
    """
    A method for SPO+ training
    """
    print("SPO+:")
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # init log
    loss_log = []
    if rlog:
        regret, _, _, _ = metric.regret(reg, optmodel, dataloader_val)
        regret_log = [regret]
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
            if rlog:
                regret, _, _, _ = metric.regret(reg, optmodel, dataloader_val)
                regret_log.append(regret)
    return loss_log, regret_log


def trainPFYL(optmodel, reg, dataloader, dataloader_val, loss_func,
              lr, num_epochs, rlog=False):
    """
    A method for PFYL training
    """
    print("PFYL:")
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # init log
    loss_log = []
    if rlog:
        regret, _, _, _ = metric.regret(reg, optmodel, dataloader_val)
        regret_log = [regret]
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
            if rlog:
                regret, _, _, _ = metric.regret(reg, optmodel, dataloader_val)
                regret_log.append(regret)
    return loss_log, regret_log


def trainNCE(optmodel, reg, dataloader, dataloader_val, loss_func,
             lr, num_epochs, rlog=False):
    """
    A method for NCE training
    """
    print("NCE:")
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # init log
    loss_log = []
    if rlog:
        regret, _, _, _ = metric.regret(reg, optmodel, dataloader_val)
        regret_log = [regret]
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
            if rlog:
                regret, _, _, _ = metric.regret(reg, optmodel, dataloader_val)
                regret_log.append(regret)
    return loss_log, regret_log
