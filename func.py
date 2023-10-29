#!/usr/bin/env python
# coding: utf-8
"""
A autograd module for cone-aligned loss
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pyepo import EPO
from pyepo.model.opt import optModel

class exactConeAlignedCosine(nn.Module):
    """
    A autograd module to align cone and vector with exact cosine similarity loss
    """

    def __init__(self, optmodel):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
        """
        super().__init__()
        # optimization model
        if not isinstance(optmodel, optModel):
            raise TypeError("arg model is not an optModel")
        self.optmodel = optmodel

    def forward(self, pred_cost, tight_ctrs, reduction="mean"):
        """
        Forward pass
        """
        loss = self._calLoss(pred_cost, tight_ctrs, self.optmodel)
        # reduction
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
        elif reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(reduction))
        return loss

    def _calLoss(self, pred_cost, tight_ctrs, optmodel):
        """
        A method to calculate loss
        """
        # get device
        device = pred_cost.device
        # get batch size
        batch_size = len(pred_cost)
        # init loss
        loss = torch.empty(batch_size).to(device)
        # cost vectors direction
        if optmodel.modelSense == EPO.MINIMIZE:
            # minimize
            pred_cost = - pred_cost
        # constraints to numpy
        tight_ctrs = tight_ctrs.cpu().detach().numpy()
        for i in range(batch_size):
            # get projection
            p = self._getProjection(pred_cost[i], tight_ctrs[i])
            # calculate cosine similarity
            loss[i] = - F.cosine_similarity(pred_cost[i].unsqueeze(0), p.unsqueeze(0))
        return loss

    def _getProjection(self, cp, ctr):
        """
        A method to get the projection of the vector onto the polar cone via solving a quadratic programming
        """
        # ceate a model
        m = gp.Model("projection")
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        p = m.addVars(len(cp), name="x", lb=-GRB.INFINITY)
        λ = m.addVars(len(ctr), name="lambda")
        # onjective function
        obj = gp.quicksum((cp[i].item() - p[i]) ** 2 for i in range(len(cp)))
        m.setObjective(obj, GRB.MINIMIZE)
        # constraints
        for i in range(len(cp)):
            m.addConstr(gp.quicksum(ctr[j,i] * λ[j] for j in range(len(ctr))) == p[i])
        # focus on numeric problem
        m.Params.NumericFocus = 3
        # solve
        m.update()
        m.optimize()
        # get solutions
        λ_val = np.array([λ[i].x for i in λ])
        # normalize
        λ_norm = λ_val / np.linalg.norm(λ_val)
        # get normalized projection
        proj = torch.FloatTensor(λ_norm @ ctr)
        return proj


class baseVectConeAlignedCosine(nn.Module):
    """
    A autograd module to align cone and vector cosine similarity loss via vase vectors
    """

    def __init__(self, optmodel):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
        """
        super().__init__()
        # optimization model
        if not isinstance(optmodel, optModel):
            raise TypeError("arg model is not an optModel")
        self.optmodel = optmodel

    def forward(self, pred_cost, tight_ctrs, reduction="mean"):
        """
        Forward pass
        """
        loss = self._calLoss(pred_cost, tight_ctrs, self.optmodel)
        # reduction
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
        elif reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(reduction))
        return loss

    def _calLoss(self, pred_cost, tight_ctrs, optmodel):
        """
        A method to calculate loss
        """
        # get device
        device = pred_cost.device
        # get batch size
        batch_size = len(pred_cost)
        # init loss
        loss = torch.empty(batch_size).to(device)
        # cost vectors direction
        if optmodel.modelSense == EPO.MINIMIZE:
            # minimize
            pred_cost = - pred_cost
        # constraints to numpy
        tight_ctrs = tight_ctrs.detach()
        for i in range(batch_size):
            # calculate cosine similarity
            loss[i] = - torch.max(F.cosine_similarity(pred_cost[i].unsqueeze(0),
                                                      tight_ctrs[i])
                        )
        return loss


class samplingConeAlignedCosine(nn.Module):
    """
    A autograd module to align cone and vector cosine similarity loss from sampling
    """

    def __init__(self, optmodel, n_samples=10):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
        """
        super().__init__()
        # optimization model
        if not isinstance(optmodel, optModel):
            raise TypeError("arg model is not an optModel")
        self.optmodel = optmodel
        self.n_samples = n_samples

    def forward(self, pred_cost, tight_ctrs, reduction="mean"):
        """
        Forward pass
        """
        loss = self._calLoss(pred_cost, tight_ctrs, self.optmodel)
        # reduction
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
        elif reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(reduction))
        return loss

    def _calLoss(self, pred_cost, tight_ctrs, optmodel):
        """
        A method to calculate loss
        """
        # get device
        device = pred_cost.device
        # get batch size
        batch_size = len(pred_cost)
        # init loss
        loss = torch.empty(batch_size).to(device)
        # cost vectors direction
        if optmodel.modelSense == EPO.MINIMIZE:
            # minimize
            pred_cost = - pred_cost
        # constraints to numpy
        tight_ctrs = tight_ctrs.cpu().detach().numpy()
        for i in range(batch_size):
            # get samples
            vecs = self._getSamples(tight_ctrs[i])
            # calculate cosine similarity
            loss[i] = - torch.max(F.cosine_similarity(pred_cost[i].unsqueeze(0),
                                                      vecs)
                        )
        return loss

    def _getSamples(self, ctr):
        """
        A method to sample vectors from rays of cone
        """
        # get solutions
        λ_val = np.random.rand(self.n_samples, len(ctr))
        # append base vectors
        #λ_val = np.concatenate(λ_val, ctr)
        # normalize
        λ_norm = λ_val / np.linalg.norm(λ_val, axis=1, keepdims=True)
        # get normalized projection
        vecs = torch.FloatTensor(λ_norm @ ctr)
        return vecs
