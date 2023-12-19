#!/usr/bin/env python
# coding: utf-8
"""
True regret loss
"""

import numpy as np
import torch
from gurobipy import GRB
from tqdm import tqdm
import pandas as pd

from pyepo import EPO

def regret(predmodel, optmodel, dataloader, skip_infeas=False):
    """
    A function to evaluate model performance with normalized true regret

    Args:
        predmodel (nn): a regression neural network for cost prediction
        optmodel (optModel): an PyEPO optimization model
        dataloader (DataLoader): Torch dataloader from optDataSet
        skip_infeas (bool): if True, skip infeasible data points

    Returns:
        float: normalized regret
        float: average number of branch-and-bound nodes
    """
    # evaluate
    predmodel.eval()
    dloss = 0 # regret
    optsum = 0
    total_node_count = 0
    num_solves = 0
    # init instance result
    regrets, mses, nodes = [], [], []
    # load data
    for data in tqdm(dataloader):
        try:
            x, c, w, z, ctr = data
        except:
            x, c, w, z = data
        # to cuda if model in cuda
        if next(predmodel.parameters()).is_cuda:
            x = x.cuda()
        # to numpy
        c = c.to("cpu").detach().numpy()
        # predict
        with torch.no_grad(): # no grad
            cp = predmodel(x).to("cpu").detach().numpy()
        # solve
        for j in range(cp.shape[0]):
            try:
                # accumulate regret
                regret = calRegret(optmodel, cp[j], c[j], z[j].item())
                dloss += regret
                regrets.append(regret)
                # accumulate mse
                mse = ((cp[j] - c[j]) ** 2).mean()
                mses.append(mse)
                # accumulate node count
                node_count = optmodel._model.getAttr(GRB.Attr.NodeCount)
                total_node_count += node_count
                nodes.append(node_count)
                # update solved number
                num_solves += 1
                # total objective value
                optsum += abs(z[j]).item()
            except AttributeError as e:
                if skip_infeas:
                    tbar.write("No feasible solution! Drop instance {}.".format(j))
                    continue  # skip this data point
                else:
                    raise ValueError("No feasible solution!")  # raise the exception
    # get tables for each instance
    instance_res = pd.DataFrame({"Regret": regrets, "MSE":mses, "Nodes": nodes})
    # turn back train mode
    predmodel.train()
    # normalized
    avg_regret = dloss / (optsum + 1e-7)
    avg_mse = instance_res["MSE"].mean()
    median_node = instance_res["Nodes"].median()
    return avg_regret, avg_mse, median_node, instance_res


def calRegret(optmodel, pred_cost, true_cost, true_obj):
    """
    A function to calculate normalized true regret for a batch

    Args:
        optmodel (optModel): optimization model
        pred_cost (torch.tensor): predicted costs
        true_cost (torch.tensor): true costs
        true_obj (torch.tensor): true optimal objective values

    Returns:predmodel
        float: true regret losses
    """
    # opt sol for pred cost
    optmodel.setObj(pred_cost)
    sol, _ = optmodel.solve()
    # obj with true cost
    obj = np.dot(sol, true_cost)
    # loss
    if optmodel.modelSense == EPO.MINIMIZE:
        loss = obj - true_obj
    if optmodel.modelSense == EPO.MAXIMIZE:
        loss = true_obj - obj
    return loss
