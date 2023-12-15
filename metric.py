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
    loss = 0
    optsum = 0
    total_node_count = 0
    num_solves = 0
    # init instance result
    regrets, nodes  = [], []
    # load data
    for data in tqdm(dataloader):
        x, c, w, z = data
        # cuda
        if next(predmodel.parameters()).is_cuda:
            x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
        # predict
        with torch.no_grad(): # no grad
            cp = predmodel(x).to("cpu").detach().numpy()
        # solve
        for j in range(cp.shape[0]):
            try:
                # accumulate loss
                regret = calRegret(optmodel, cp[j], c[j].to("cpu").detach().numpy(),
                              z[j].item())
                loss += regret
                regrets.append(regret)
                # accumulate node count
                node_count = optmodel._model.getAttr(GRB.Attr.NodeCount)
                total_node_count += node_count
                nodes.append(node_count)
                # update solved number
                num_solves += 1
            except AttributeError as e:
                if self.skip_infeas:
                    tbar.write("No feasible solution! Drop instance {}.".format(j))
                    continue  # skip this data point
                else:
                    raise ValueError("No feasible solution!")  # raise the exception
        optsum += abs(z).sum().item()
    # get tables for each instance
    instance_res = pd.DataFrame({"Regret": regrets, "Nodes": nodes})
    # turn back train mode
    predmodel.train()
    # normalized
    return loss / (optsum + 1e-7), instance_res["Nodes"].median(), instance_res


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
