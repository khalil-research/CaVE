#!/usr/bin/env python
# coding: utf-8
"""
A autograd module for cone-aligned loss
"""

from abc import ABC, abstractmethod

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.optimize import nnls
#from fnnls import fnnls
import torch
from torch import nn
from torch.nn import functional as F

from pyepo import EPO
from pyepo.model.opt import optModel

class abstractConeAlignedCosine(nn.Module, ABC):
    """
    Abstract base class for cone-aligned cosine loss modules.
    """
    def __init__(self, optmodel):
        """
        Initialize the abstract class with an optimization model.
        Args:
            optmodel (optModel): an PyEPO optimization model
        """
        super().__init__()
        if not isinstance(optmodel, optModel):
            raise TypeError("arg model is not an optModel")
        self.optmodel = optmodel

    def forward(self, pred_cost, tight_ctrs, reduction="mean"):
        """
        Forward pass method.
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

    @abstractmethod
    def _calLoss(self, pred_cost, tight_ctrs, optmodel):
        """
        Abstract method to calculate loss.
        """
        pass

    def _checkInCone(self, cp, ctr):
        """
        Method to check if the given cost vector in the cone
        """
        # drop pads
        ctr = ctr[np.abs(ctr).sum(axis=1) > 1e-5]
        # ceate a model
        m = gp.Model("Cone Combination")
        # turn off output
        m.Params.outputFlag = 0
        # numerical precision
        m.Params.FeasibilityTol = 1e-3
        m.Params.OptimalityTol = 1e-3
        # varibles
        λ = m.addMVar(len(ctr), name="λ")
        # constraints
        m.addConstr(λ @ ctr == cp)
        # objective function
        m.setObjective(0, GRB.MINIMIZE)
        # solve the model
        m.optimize()
        # return the status of the model
        return m.status == GRB.OPTIMAL


class exactConeAlignedCosine(abstractConeAlignedCosine):
    """
    A autograd module to align cone and vector with exact cosine similarity loss
    """
    def __init__(self, optmodel, warmstart=True, conecheck=False):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
        """
        super().__init__(optmodel)
        self.warmstart = warmstart
        self.conecheck = conecheck

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
        # to numpy
        cp = pred_cost.detach().cpu().numpy()
        ctrs = tight_ctrs.detach().cpu().numpy()
        for i in range(batch_size):
            if self.conecheck and self._checkInCone(cp[i], ctrs[i]):
                # in the cone
                p = torch.FloatTensor(cp[i].copy())
            else:
                # get projection
                p = self._getProjection(cp[i], ctrs[i])
            # calculate cosine similarity
            loss[i] = - F.cosine_similarity(pred_cost[i].unsqueeze(0),
                                            p.unsqueeze(0))
        return loss

    def _getProjection(self, cp, ctr):
        """
        A method to get the projection of the vector onto the polar cone via solving a quadratic programming
        """
        # drop pads
        ctr = ctr[np.abs(ctr).sum(axis=1) > 1e-5]
        # ceate a model
        m = gp.Model("projection")
        # turn off output
        m.Params.outputFlag = 0
        # numerical precision
        m.Params.FeasibilityTol = 1e-3
        m.Params.OptimalityTol = 1e-3
        # varibles
        λ = m.addMVar(len(ctr), name="λ")
        # warm-start
        if self.warmstart:
            init_λ = np.zeros(len(ctr))
            sign = ctr[-len(cp):].sum(axis=0)
            init_λ[-len(cp):] = np.abs(cp) * (np.sign(cp) == np.sign(sign)) # hyperqudrants
            λ.Start = init_λ
        # objective function
        obj = (cp - λ @ ctr) @ (cp - λ @ ctr)
        m.setObjective(obj, GRB.MINIMIZE)
        # focus on numeric problem
        m.Params.NumericFocus = 3
        # solve
        m.optimize()
        # get solutions
        proj = np.array(λ.X) @ ctr
        # normalize
        proj = torch.FloatTensor(proj / np.linalg.norm(proj))
        return proj


class nnlsConeAlignedCosine(abstractConeAlignedCosine):
    """
    A autograd module to align cone and vector with non-negative least squares
    """
    def __init__(self, optmodel, conecheck=False):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
        """
        super().__init__(optmodel)
        self.conecheck = conecheck

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
        # to numpy
        cp = pred_cost.detach().cpu().numpy()
        ctrs = tight_ctrs.detach().cpu().numpy()
        for i in range(batch_size):
            if self.conecheck and self._checkInCone(cp[i], ctrs[i]):
                # in the cone
                p = torch.FloatTensor(cp[i].copy())
            else:
                # get projection
                p = self._getProjection(cp[i], ctrs[i])
            # calculate cosine similarity
            loss[i] = - F.cosine_similarity(pred_cost[i].unsqueeze(0),
                                            p.unsqueeze(0))
        return loss

    def _getProjection(self, cp, ctr):
        # drop pads
        ctr = ctr[np.abs(ctr).sum(axis=1) > 1e-5]
        # solve the linear equations
        λ, _ = nnls(ctr.T, cp)
        #λ, _ = fnnls(ctr.T, cp, epsilon=1e-5)
        # get projection
        proj = λ @ ctr
        # normalize
        proj = proj / np.linalg.norm(proj)
        return torch.FloatTensor(proj)


class samplingConeAlignedCosine(abstractConeAlignedCosine):
    """
    A autograd module to align cone and vector cosine similarity loss from sampling
    """
    def __init__(self, optmodel, n_samples=10):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
        """
        super().__init__(optmodel)
        self.n_samples = n_samples

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
        # get samples
        vecs = self._getSamples(tight_ctrs.cpu().detach().numpy())
        # calculate cosine similarity
        cos_sim = F.cosine_similarity(pred_cost.unsqueeze(1), vecs, dim=2)
        # get max cosine similarity for each sample
        max_cos_sim, _ = torch.max(cos_sim, dim=1)
        loss = - max_cos_sim
        return loss

    def _getSamples(self, ctrs):
        """
        A method to sample vectors from rays of cone
        """
        # get solutions
        λ_val = np.random.rand(ctrs.shape[0], self.n_samples, ctrs.shape[1])
        # normalize
        λ_norm = λ_val / np.linalg.norm(λ_val, axis=2, keepdims=True)
        # get normalized projection
        vecs = torch.FloatTensor(λ_norm @ ctrs)
        return vecs


class signConeAlignedCosine(abstractConeAlignedCosine):
    """
    A autograd module to quickly align vector to the subset (hyperquadrant) of cone cosine similarity loss
    """
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
        # to numpy
        cp = pred_cost.detach().cpu().numpy()
        ctrs = tight_ctrs.detach().cpu().numpy()
        for i in range(batch_size):
            # get projection
            p = self._getSignProjection(cp[i], ctrs[i])
            # calculate cosine similarity
            loss[i] = - F.cosine_similarity(pred_cost[i].unsqueeze(0),
                                            p.unsqueeze(0))
        return loss

    def _getSignProjection(self, cp, ctr):
        """
        A method to get the projection of the vector onto the hyperquadrant cone
        """
        sign = ctr[-len(cp):].sum(axis=0)
        # get projection on the hyperquadrant
        proj = cp * (np.sign(cp) == np.sign(sign))
        # normalize
        proj = torch.FloatTensor(proj / np.linalg.norm(proj))
        return proj
