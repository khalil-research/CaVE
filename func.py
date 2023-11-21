#!/usr/bin/env python
# coding: utf-8
"""
An autograd module for cone-aligned loss
"""

from abc import ABC, abstractmethod

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from pathos.multiprocessing import ProcessingPool
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

    def _calLoss(self, pred_cost, tight_ctrs, optmodel):
        """
        Abstract method to calculate loss.
        """
        # cost vectors direction
        if optmodel.modelSense == EPO.MINIMIZE:
            # minimize
            pred_cost = - pred_cost
        # get projection
        proj = self._getProjection(pred_cost, tight_ctrs)
        # calculate cosine similarity
        loss = - F.cosine_similarity(pred_cost, proj, dim=1)
        return loss

    @abstractmethod
    def _getProjection(self, pred_cost, tight_ctrs):
        """
        Abstract method to obtain projection.
        """
        pass

def _checkInCone(cp, ctr):
    """
    Method to check if the given cost vector in the cone
    """
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
    def __init__(self, optmodel, warmstart=True, conecheck=False, processes=1):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            warmstart (bool): start QP with initial solutions or not
            conecheck (bool): check if cost vector is in the cone or not
            processes (int): number of processors, 1 for single-core, 0 for all of cores
        """
        super().__init__(optmodel)
        self.warmstart = warmstart
        self.conecheck = conecheck
        self.processes = mp.cpu_count() if not processes else processes
        # single-core
        if self.processes == 1:
            self.pool = None
        # multi-core
        else:
            self.pool = ProcessingPool(self.processes)
        print("Num of cores: {}".format(self.processes))

    def _getProjection(self, pred_cost, tight_ctrs):
        """
        A method to get the projection of the vector onto the polar cone via solving a quadratic programming
        """
        # get device
        device = pred_cost.device
        # to numpy
        pred_cost = pred_cost.detach().cpu().numpy()
        tight_ctrs = tight_ctrs.detach().cpu().numpy()
        # single-core
        if self.processes == 1:
            # init loss
            proj = torch.empty(pred_cost.shape).to(device)
            # calculate projection
            for i, (cp, ctr) in enumerate(zip(pred_cost, tight_ctrs)):
                proj[i] = self._solveQP(cp, ctr, self.warmstart, self.conecheck)
        # multi-core
        else:
            res = self.pool.amap(self._solveQP,
                                 pred_cost, tight_ctrs,
                                 [self.warmstart]*len(pred_cost),
                                 [self.conecheck]*len(pred_cost)).get()
            proj = torch.stack(res, dim=0)
        return proj

    @staticmethod
    def _solveQP(cp, ctr, warmstart, conecheck):
        """
        A static method to solve quadratic programming.
        """
        # drop pads
        ctr = ctr[np.abs(ctr).sum(axis=1) > 1e-7]
        # check if in the cone
        if conecheck and _checkInCone(cp, ctr):
            return torch.FloatTensor(cp.copy())
        # ceate a model
        m = gp.Model("projection")
        # turn off output
        m.Params.outputFlag = 0
        # numerical precision
        m.Params.FeasibilityTol = 1e-3
        m.Params.OptimalityTol = 1e-3
        # focus on numeric problem
        m.Params.NumericFocus = 3
        # varibles
        λ = m.addMVar(len(ctr), name="λ")
        # warm-start
        if warmstart:
            init_λ = np.zeros(len(ctr))
            sign = ctr[-len(cp):].sum(axis=0)
            init_λ[-len(cp):] = np.abs(cp) * (np.sign(cp) == np.sign(sign)) # hyperqudrants
            λ.Start = init_λ
        # objective function
        obj = (cp - λ @ ctr) @ (cp - λ @ ctr)
        m.setObjective(obj, GRB.MINIMIZE)
        # solve
        m.optimize()
        # get solutions
        p = np.array(λ.X) @ ctr
        # normalize
        p = torch.FloatTensor(p / np.linalg.norm(p))
        return p


class nnlsConeAlignedCosine(abstractConeAlignedCosine):
    """
    A autograd module to align cone and vector with non-negative least squares
    """
    def __init__(self, optmodel, conecheck=False, processes=1):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            conecheck (bool): check if cost vector is in the cone or not
            processes (int): number of processors, 1 for single-core, 0 for all of cores
        """
        super().__init__(optmodel)
        self.conecheck = conecheck
        self.processes = mp.cpu_count() if not processes else processes
        # single-core
        if self.processes == 1:
            self.pool = None
        # multi-core
        else:
            self.pool = ProcessingPool(self.processes)
        print("Num of cores: {}".format(self.processes))

    def _getProjection(self, pred_cost, tight_ctrs):
        """
        A method to get the projection of the vector onto the polar cone via solving a quadratic programming
        """
        # get device
        device = pred_cost.device
        # to numpy
        pred_cost = pred_cost.detach().cpu().numpy()
        tight_ctrs = tight_ctrs.detach().cpu().numpy()
        # single-core
        if self.processes == 1:
            # init loss
            proj = torch.empty(pred_cost.shape).to(device)
            # calculate projection
            for i, (cp, ctr) in enumerate(zip(pred_cost, tight_ctrs)):
                proj[i] = self._solveNNLS(cp, ctr, self.conecheck)
        # multi-core
        else:
            res = self.pool.amap(self._solveNNLS,
                                 pred_cost, tight_ctrs,
                                 [self.conecheck]*len(pred_cost)).get()
            proj = torch.stack(res, dim=0)
        return proj

    @staticmethod
    def _solveNNLS(cp, ctr, conecheck):
        # drop pads
        ctr = ctr[np.abs(ctr).sum(axis=1) > 1e-7]
        # check if in the cone
        if conecheck and _checkInCone(cp, ctr):
            return torch.FloatTensor(cp.copy())
        # solve the linear equations
        λ, _ = nnls(ctr.T, cp)
        #λ, _ = fnnls(ctr.T, cp, epsilon=1e-5)
        # get projection
        proj = λ @ ctr
        # normalize
        proj = proj / np.linalg.norm(proj)
        return torch.FloatTensor(proj)


class avgConeAlignedCosine(abstractConeAlignedCosine):
    """
    A autograd module to align cone and vector cosine similarity loss via average base vectors
    """
    def _getProjection(self, pred_cost, tight_ctrs):
        """
        A method to average of base vectors
        """
        return tight_ctrs.mean(dim=1)


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
        # cost vectors direction
        if optmodel.modelSense == EPO.MINIMIZE:
            # minimize
            pred_cost = - pred_cost
        # get samples
        vecs = self._getProjection(tight_ctrs.cpu().detach().numpy())
        # calculate cosine similarity
        cos_sim = F.cosine_similarity(pred_cost.unsqueeze(1), vecs, dim=2)
        # get max cosine similarity for each sample
        max_cos_sim, _ = torch.max(cos_sim, dim=1)
        loss = - max_cos_sim
        return loss

    def _getProjection(self, tight_ctrs):
        """
        A method to sample vectors from rays of cone
        """
        # get solutions
        λ_val = np.random.rand(tight_ctrs.shape[0],
                               self.n_samples,
                               tight_ctrs.shape[1])
        # normalize
        λ_norm = λ_val / np.linalg.norm(λ_val, axis=2, keepdims=True)
        # get normalized projection
        vecs = torch.FloatTensor(λ_norm @ tight_ctrs)
        return vecs.detach()


class signConeAlignedCosine(abstractConeAlignedCosine):
    """
    A autograd module to quickly align vector to the subset (hyperquadrant) of cone cosine similarity loss
    """
    def _getProjection(self, pred_cost, tight_ctrs):
        """
        A method to get the projection of the vector onto the hyperquadrant cone
        """
        # compute signs
        sign = tight_ctrs[:, -pred_cost.shape[1]:].sum(dim=1)
        # get projection on the hyperquadrant
        proj = pred_cost * (torch.sign(pred_cost) == torch.sign(sign))
        # normalize
        norm = torch.linalg.norm(proj, dim=1, keepdim=True)
        proj = proj / norm
        return proj.detach()
