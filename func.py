#!/usr/bin/env python
# coding: utf-8
"""
An autograd module for cone-aligned loss
"""

from abc import ABC, abstractmethod

import cvxpy as cvx
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
    def __init__(self, optmodel, reduction="mean", processes=1):
        """
        Initialize the abstract class with an optimization model.
        Args:
            optmodel (optModel): an PyEPO optimization model
            reduction (str): the reduction to apply to the output
            processes (int): number of processors, 1 for single-core, 0 for all of coress
        """
        super().__init__()
        if not isinstance(optmodel, optModel):
            raise TypeError("arg model is not an optModel")
        self.optmodel = optmodel
        self.reduction = reduction
        # number of processes
        self.processes = mp.cpu_count() if not processes else processes
        # single-core
        if self.processes == 1:
            self.pool = None
        # multi-core
        else:
            self.pool = ProcessingPool(self.processes)
        print("Num of cores: {}".format(self.processes))

    def forward(self, pred_cost, tight_ctrs):
        """
        Forward pass method.
        """
        loss = self._calLoss(pred_cost, tight_ctrs, self.optmodel)
        # reduction
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(reduction))
        return loss

    def _calLoss(self, pred_cost, tight_ctrs, optmodel):
        """
        A method to calculate loss.
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


class exactConeAlignedCosine(abstractConeAlignedCosine):
    """
    A autograd module to align cone and vector with exact cosine similarity loss
    """
    def __init__(self, optmodel, solver=None, reduction="mean", processes=1):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            solver (str): the QP solver to find projection
            reduction (str): the reduction to apply to the output
            processes (int): number of processors, 1 for single-core, 0 for all of coress
        """
        super().__init__(optmodel, reduction, processes)
        # solver
        if solver == "gurobi":
            self._solveQP = self._solveGurobi
        if solver == "clarabel":
            self._solveQP = self._solveClarabel
        if solver == "nnls":
            self._solveQP = self._solveNNLS

    def _getProjection(self, pred_cost, tight_ctrs):
        """
        A method to get the projection of the vector onto the polar cone via solving a quadratic programming
        """
        # get device
        device = pred_cost.device
        # single-core
        if self.processes == 1:
            # init loss
            proj = torch.empty(pred_cost.shape).to(device)
            # calculate projection
            for i, (cp, ctr) in enumerate(zip(pred_cost, tight_ctrs)):
                proj[i], _ = self._solveQP(cp, ctr)
        # multi-core
        else:
            res = self.pool.amap(self._solveQP, pred_cost, tight_ctrs).get()
            proj, _ = zip(*res)
            proj = torch.stack(proj, dim=0).to(device)
        # normalize
        vec = proj / proj.norm(dim=1, keepdim=True)
        return vec

    @staticmethod
    def _solveQP(cp, ctr):
        """
        A unimplemented method requires to solve QP
        """
        raise ValueError("No solver and its corresponding '_solveQP' method.")

    @staticmethod
    def _solveGurobi(cp, ctr):
        """
        A static method to solve quadratic programming with gurobi
        """
        # to numpy
        cp = cp.detach().cpu().numpy()
        ctr = ctr.detach().cpu().numpy()
        # drop pads
        ctr = ctr[np.abs(ctr).sum(axis=1) > 1e-7]
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
        # objective function
        obj = (cp - λ @ ctr) @ (cp - λ @ ctr)
        m.setObjective(obj, GRB.MINIMIZE)
        # solve
        m.optimize()
        # get solutions
        p = λ.X @ ctr
        # get value
        rnorm = m.ObjVal
        return torch.FloatTensor(p), rnorm

    @staticmethod
    def _solveClarabel(cp, ctr):
        """
        A static method to solve quadratic programming with Clarabel
        """
        # to numpy
        cp = cp.detach().cpu().numpy()
        ctr = ctr.detach().cpu().numpy()
        # drop pads
        ctr = ctr[np.abs(ctr).sum(axis=1) > 1e-7]
        # varibles
        λ = cvx.Variable(len(ctr), name="λ", nonneg=True)
        # onjective function
        objective = cvx.Minimize(cvx.sum_squares(cp - λ @ ctr))
        # ceate a model
        problem = cvx.Problem(objective)
        # solve and focus on numeric problem
        problem.solve(solver=cvx.CLARABEL,
                      tol_infeas_rel=1e-3, tol_feas=1e-3, max_iter=20)
        # get solutions
        p = λ.value @ ctr
        # get value
        rnorm = problem.value
        return torch.FloatTensor(p), rnorm

    @staticmethod
    def _solveNNLS(cp, ctr):
        """
        A static method to solve quadratic programming with scipy
        """
        # to numpy
        cp = cp.detach().cpu().numpy()
        ctr = ctr.detach().cpu().numpy()
        # drop pads
        ctr = ctr[np.abs(ctr).sum(axis=1) > 1e-7]
        # solve the linear equations
        λ, rnorm = nnls(ctr.T, cp)
        #λ, _ = fnnls(ctr.T, cp, epsilon=1e-5)
        # get projection
        p = λ @ ctr
        return torch.FloatTensor(p), rnorm


class innerConeAlignedCosine(exactConeAlignedCosine):
    def __init__(self, optmodel, solver=None, solve_ratio=1,
                 inner_ratio=0.2, reduction="mean", processes=1):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            solver (str): the QP solver to find projection
            solve_ratio (float): the ratio of solving QP during training
            inner_ratio (float): the ratio to push projection inside
            reduction (str): the reduction to apply to the output
            processes (int): number of processors, 1 for single-core, 0 for all of coress
        """
        super().__init__(optmodel, solver, reduction, processes)
        # solve ratio
        self.solve_ratio = solve_ratio
        if (self.solve_ratio < 0) or (self.solve_ratio > 1):
            raise ValueError("Invalid solving ratio {}. It should be between 0 and 1.".
                format(self.solve_ratio))
        # inner ratio
        self.inner_ratio = inner_ratio
        if (self.inner_ratio < 0) or (self.inner_ratio > 1):
            raise ValueError("Invalid inner ratio {}. It should be between 0 and 1.".
                format(self.inner_ratio))

    def _getProjection(self, pred_cost, tight_ctrs):
        """
        A method to get the projection of the vector onto the polar cone via solving a quadratic programming
        """
        # get device
        device = pred_cost.device
        # solve QP
        if np.random.uniform() <= self.solve_ratio:
             # single-core
            if self.processes == 1:
                # init loss
                proj = torch.empty(pred_cost.shape).to(device)
                rnorm = torch.empty(pred_cost.shape[0]).to(device)
                # calculate projection
                for i, (cp, ctr) in enumerate(zip(pred_cost, tight_ctrs)):
                    proj[i], rnorm[i] = self._solveQP(cp, ctr)
            # multi-core
            else:
                res = self.pool.amap(self._solveQP, pred_cost, tight_ctrs).get()
                proj, rnorm = zip(*res)
                proj = torch.stack(proj, dim=0).to(device)
                rnorm = torch.tensor(rnorm).to(device)
            # normalize
            proj = proj / proj.norm(dim=1, keepdim=True)
            # get average
            avg = self._getAvg(tight_ctrs)
            # combine vector
            vec = (1 - self.inner_ratio) * proj + self.inner_ratio * avg
            # projection is itself if in the cone
            vec[rnorm < 1e-7] = proj[rnorm < 1e-7]
        # fake projection
        else:
            # get average
            avg = self._getAvg(tight_ctrs)
            # normalize
            pred_norm = pred_cost / pred_cost.norm(dim=1, keepdim=True)
            # combine vector
            vec = (1 - self.inner_ratio) * pred_norm + self.inner_ratio * avg
        return vec.detach()

    def _getAvg(self, tight_ctrs):
        """
        A method to get average of binding constraints
        """
        # normalize
        tight_ctrs = tight_ctrs / (tight_ctrs.norm(dim=2, keepdim=True) + 1e-8)
        return tight_ctrs.mean(dim=1).detach()


class avgConeAlignedCosine(abstractConeAlignedCosine):
    """
    A autograd module to align cone and vector cosine similarity loss via average base vectors
    """
    def __init__(self, optmodel, check_cone=False, inner_ratio=0.2,
                 reduction="mean", processes=1):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            check_cone (bool): if check the cost vector in the cone or not
            inner_ratio (float): the ratio to push projection inside
            reduction (str): the reduction to apply to the output
            processes (int): number of processors, 1 for single-core, 0 for all of cores
        """
        super().__init__(optmodel, reduction, processes)
        self.check_cone = check_cone
        # inner ratio
        self.inner_ratio = inner_ratio
        if (self.inner_ratio < 0) or (self.inner_ratio > 1):
            raise ValueError("Invalid inner ratio {}. It should be between 0 and 1.".
                format(self.inner_ratio))

    def _getProjection(self, pred_cost, tight_ctrs):
        """
        A method to get average of binding constraints
        """
        # normalize
        tight_ctrs = tight_ctrs / (tight_ctrs.norm(dim=2, keepdim=True) + 1e-8)
        # get average
        avg = tight_ctrs.mean(dim=1).detach()
        # normalize
        pred_norm = pred_cost / pred_cost.norm(dim=1, keepdim=True)
        # combine vector
        vec = (1 - self.inner_ratio) * pred_norm + self.inner_ratio * avg
        # cone check
        if self.check_cone:
            # update projection
            _updateProjectionIfInCone(vecs, pred_cost, tight_ctrs,
                                      self.processes, self.pool)
        return vec.detach()


class samplingConeAlignedCosine(abstractConeAlignedCosine):
    """
    A autograd module to align cone and vector cosine similarity loss from sampling
    """
    def __init__(self, optmodel, n_samples=10, check_cone=False, inner_ratio=0.2,
                 reduction="mean", processes=1):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            n_samples (int): number of samples
            check_cone (bool): if check the cost vector in the cone or not
            inner_ratio (float): the ratio to push projection inside
            reduction (str): the reduction to apply to the output
            processes (int): number of processors, 1 for single-core, 0 for all of cores
        """
        super().__init__(optmodel, reduction, processes)
        self.n_samples = n_samples
        self.check_cone = check_cone
        # inner ratio
        self.inner_ratio = inner_ratio
        if (self.inner_ratio < 0) or (self.inner_ratio > 1):
            raise ValueError("Invalid inner ratio {}. It should be between 0 and 1.".
                format(self.inner_ratio))

    def _calLoss(self, pred_cost, tight_ctrs, optmodel):
        """
        A method to calculate loss
        """
        # cost vectors direction
        if optmodel.modelSense == EPO.MINIMIZE:
            # minimize
            pred_cost = - pred_cost
        # get samples
        samples = self._getProjection(pred_cost, tight_ctrs)
        # normalize
        pred_norm = (pred_cost / pred_cost.norm(dim=1, keepdim=True)).unsqueeze(1)
        # combine vector
        vecs = (1 - self.inner_ratio) * pred_norm + self.inner_ratio * samples
        vecs = vecs.detach()
        # calculate cosine similarity
        cos_sim = F.cosine_similarity(pred_cost.unsqueeze(1), vecs, dim=2)
        # get max cosine similarity for each sample
        max_cos_sim, _ = torch.max(cos_sim, dim=1)
        loss = - max_cos_sim
        return loss

    def _getProjection(self, pred_cost, tight_ctrs):
        """
        A method to sample vectors from rays of cone
        """
        # get device
        device = tight_ctrs.device
        # normalize constraints
        tight_ctrs = tight_ctrs / (tight_ctrs.norm(dim=2, keepdim=True) + 1e-8)
        # random weights
        λ_val = torch.rand(tight_ctrs.shape[0], self.n_samples, tight_ctrs.shape[1])
        λ_val = λ_val.to(device)
        λ_val = λ_val / λ_val.sum(dim=2, keepdims=True)
        # get projection
        vecs = λ_val @ tight_ctrs
        # cone check
        if self.check_cone:
            # update projection
            _updateProjectionIfInCone(vecs, pred_cost, tight_ctrs,
                                      self.processes, self.pool)
        return vecs.detach()


class signConeAlignedCosine(abstractConeAlignedCosine):
    """
    A autograd module to quickly align vector to the subset (hyperquadrant) of cone cosine similarity loss
    """
    def __init__(self, optmodel, check_cone=False, inner_ratio=0.2,
                 reduction="mean", processes=1):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            check_cone (bool): if check the cost vector in the cone or not
            inner_ratio (float): the ratio to push projection inside
            reduction (str): the reduction to apply to the output
            processes (int): number of processors, 1 for single-core, 0 for all of cores
        """
        super().__init__(optmodel, reduction, processes)
        self.check_cone = check_cone
        # inner ratio
        self.inner_ratio = inner_ratio
        if (self.inner_ratio < 0) or (self.inner_ratio > 1):
            raise ValueError("Invalid inner ratio {}. It should be between 0 and 1.".
                format(self.inner_ratio))

    def _getProjection(self, pred_cost, tight_ctrs):
        """
        A method to get the projection of the vector onto the hyperquadrant cone
        """
        # compute signs
        sign = tight_ctrs[:, -pred_cost.shape[1]:].sum(dim=1)
        # get projection on the hyperquadrant
        proj = pred_cost * (torch.sign(pred_cost) == torch.sign(sign))
        # normalize
        proj = proj / proj.norm(dim=1, keepdim=True)
        # normalize
        pred_norm = pred_cost / pred_cost.norm(dim=1, keepdim=True)
        # combine vector
        vec = (1 - self.inner_ratio) * pred_norm + self.inner_ratio * proj
        # cone check
        if self.check_cone:
            # update projection
            _updateProjectionIfInCone(proj, pred_cost, tight_ctrs,
                                      self.processes, self.pool)
        return vec.detach()


def _updateProjectionIfInCone(proj, pred_cost, tight_ctrs, processes, pool):
    """
    A method to update projection of cost vectr in cone as itself
    """
    # single-core
    if processes == 1:
        for i, (cp, ctr) in enumerate(zip(pred_cost, tight_ctrs)):
            # if in the cone
            if _checkInCone(cp, ctr):
                # projection is itself
                proj[i] = cp
    # multi-core
    else:
        # check if in the cone
        res = pool.amap(_checkInCone, pred_cost, tight_ctrs).get()
        # projection is itself
        proj[res] = pred_cost[res]


def _checkInCone(cp, ctr):
    """
    Method to check if the given cost vector in the cone
    """
    # to numpy
    cp = cp.detach().cpu().numpy()
    ctr = ctr.detach().cpu().numpy()
    # drop pads
    ctr = ctr[np.abs(ctr).sum(axis=1) > 1e-7]
    # ceate a model
    m = gp.Model("Cone Combination")
    # turn off output
    m.Params.outputFlag = 0
    # numerical precision
    m.Params.FeasibilityTol = 1e-3
    m.Params.OptimalityTol = 1e-3
    # variables
    λ = m.addMVar(len(ctr), name="λ")
    # constraints
    m.addConstr(λ @ ctr == cp)
    # objective function
    m.setObjective(0, GRB.MINIMIZE)
    # solve the model
    m.optimize()
    # return the status of the model
    return m.status == GRB.OPTIMAL
