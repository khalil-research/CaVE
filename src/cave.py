#!/usr/bin/env python
# coding: utf-8
"""
An autograd module for cone-aligned loss
"""

from abc import ABC, abstractmethod

import cvxpy as cvx
import numpy as np
from pathos.multiprocessing import ProcessingPool
from scipy.optimize import nnls
import torch
from torch import nn
from torch.nn import functional as F

from pyepo import EPO
from pyepo.model.opt import optModel

class abstractConeAlignedCosine(nn.Module, ABC):
    """
    An abstract base class for CaVE loss.
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
        # cost vectors direction
        if optmodel.modelSense == EPO.MINIMIZE:
            # set vector sign to -1 for minimization
            self.vec_sign = -1
        else:
            # set vector sign to 1 for maximization
            self.vec_sign = 1
        # method for aggregating the loss
        self.reduction = reduction
        if self.reduction not in ["mean", "sum", "none"]:
            message = ValueError("No reduction '{}'.".format(self.reduction))
        # number of processes
        self.processes = mp.cpu_count() if not processes else processes
        # multi-core
        if self.processes > 1:
            # create a processes pool
            self.pool = ProcessingPool(self.processes)
        print("Num of cores: {}".format(self.processes))

    def forward(self, pred_cost, tight_ctrs):
        """
        A Forward pass method.
        """
        loss = self._calLoss(pred_cost, tight_ctrs)
        # loss reduction
        if self.reduction == "mean":
            loss = torch.mean(loss)
        if self.reduction == "sum":
            loss = torch.sum(loss)
        if self.reduction == "none":
            loss = loss
        return loss

    def _calLoss(self, pred_cost, tight_ctrs):
        """
        A method to calculate loss.
        """
        # change cost vectors direction
        pred_cost = self.vec_sign * pred_cost
        # get projection
        proj = self._getProjection(pred_cost, tight_ctrs)
        # calculate cosine similarity between predicted costs and their projections
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
    An autograd module for CaVE Exact
    """
    def __init__(self, optmodel, solver="clarabel", reduction="mean", processes=1):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            solver (str): the QP solver to find projection
            reduction (str): the reduction to apply to the output
            processes (int): number of processors, 1 for single-core, 0 for all of coress
        """
        super().__init__(optmodel, reduction, processes)
        # choose between 'clarabel' or 'nnls' as the QP solver.
        self.solver = solver
        if self.solver not in ["clarabel", "nnls"]:
            message = "Invalid solver: {}. Must be 'clarabel' or 'nnls'.".format(self.solver)
            raise ValueError(message)
        # clarabel which has better scalability
        if self.solver == "clarabel":
            self._solveQP = self._solveClarabel
        # nnls from scipy which is very fast for small problems
        if self.solver == "nnls":
            self._solveQP = self._solveNNLS

    def _getProjection(self, pred_cost, tight_ctrs):
        """
        A method to get the exact projection onto the optimal subcone.
        """
        # get device for output tensor placement
        device = pred_cost.device
        # to numpy
        pred_cost = pred_cost.detach().cpu().numpy()
        tight_ctrs = tight_ctrs.detach().cpu().numpy()
        # single-core
        if self.processes == 1:
            # init empty tensor
            proj = torch.empty(pred_cost.shape).to(device)
            # calculate projections per instance
            for i, (cp, ctr) in enumerate(zip(pred_cost, tight_ctrs)):
                # solve QP
                proj[i], _ = self._solveQP(cp, ctr)
        # multi-core
        else:
            # calculate projections with pool
            res = self.pool.amap(self._solveQP, pred_cost, tight_ctrs).get()
            # the projection
            proj, _ = zip(*res)
            proj = torch.stack(proj, dim=0).to(device)
        # normalize
        # QUESTION: Do we need to normalize it?
        vec = proj / proj.norm(dim=1, keepdim=True)
        return vec

    @staticmethod
    def _solveQP(cp, ctr):
        """
        A unimplemented method requires to solve QP
        """
        raise ValueError("No solver and its corresponding '_solveQP' method.")

    @staticmethod
    def _solveClarabel(cp, ctr):
        """
        A static method to solve quadratic programming with Clarabel
        """
        # remove zero-padding from binding constraints
        ctr = ctr[np.abs(ctr).sum(axis=1) > 1e-7]
        # decision varibles λ
        λ = cvx.Variable(len(ctr), name="λ", nonneg=True)
        # onjective function ||cp - λ ctr||^2
        objective = cvx.Minimize(cvx.sum_squares(cp - λ @ ctr))
        # ceate an optimization problem model
        problem = cvx.Problem(objective)
        # solve with Clarabel
        problem.solve(solver=cvx.CLARABEL)
        # obtain the closest projection on the surface of the cone
        p = λ.value @ ctr
        # compute residuals as the Euclidean distance between prediction and projection
        rnorm = problem.value
        return torch.tensor(p, dtype=torch.float32), rnorm

    @staticmethod
    def _solveNNLS(cp, ctr):
        """
        A static method to solve quadratic programming with scipy
        """
        # remove zero-padding from binding constraints
        ctr = ctr[np.abs(ctr).sum(axis=1) > 1e-7]
        # solve as non-negaitive least square (using the active set method)
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html
        λ, rnorm = nnls(ctr.T, cp)
        # compute residuals as the Euclidean distance between prediction and projection
        p = λ @ ctr
        return torch.tensor(p, dtype=torch.float32), rnorm


class innerConeAlignedCosine(exactConeAlignedCosine):
    """
    An autograd module for CaVE+ and CaVE Hybrid.
    """
    def __init__(self, optmodel, solver="clarabel", max_iter=3, solve_ratio=1,
                 inner_ratio=0.2, reduction="mean", processes=1):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            solver (str): the QP solver to find projection
            max_iter (int): the maximum number of iterations
            solve_ratio (float): the ratio of solving QP during training
            inner_ratio (float): the weight to push heuristic projection inside
            reduction (str): the reduction to apply to the output
            processes (int): number of processors, 1 for single-core, 0 for all of coress
        """
        super().__init__(optmodel, solver, reduction, processes)
        # maximum iterations
        self.max_iter = max_iter
        # solve ratio
        self.solve_ratio = solve_ratio
        # check if value is valid [0,1]
        if (self.solve_ratio < 0) or (self.solve_ratio > 1):
            raise ValueError("Invalid solving ratio {}. It should be between 0 and 1.".
                format(self.solve_ratio))
        # inner ratio
        self.inner_ratio = inner_ratio
        # check if value is valid [0,1]
        if (self.inner_ratio < 0) or (self.inner_ratio > 1):
            raise ValueError("Invalid inner ratio {}. It should be between 0 and 1.".
                format(self.inner_ratio))

    def _getProjection(self, pred_cost, tight_ctrs):
        """
        A method to get the inner projection inside the optimal subcone.
        """
        # get device
        device = pred_cost.device
        # get average of the (normalized) binding constraints
        avg = self._getAvg(tight_ctrs)
        # inner project with QP
        if np.random.uniform() <= self.solve_ratio:
            # to numpy
            pred_cost = pred_cost.detach().cpu().numpy()
            tight_ctrs = tight_ctrs.detach().cpu().numpy()
             # single-core
            if self.processes == 1:
                # init empty tensor
                proj = torch.empty(pred_cost.shape).to(device)
                rnorm = torch.empty(pred_cost.shape[0]).to(device)
                # calculate projections per instance
                for i, (cp, ctr) in enumerate(zip(pred_cost, tight_ctrs)):
                    # solve QP
                    proj[i], rnorm[i] = self._solveQP(cp, ctr, self.max_iter)
            # multi-core
            else:
                # calculate projections with pool
                res = self.pool.amap(self._solveQP, pred_cost, tight_ctrs,
                                     [self.max_iter]*len(pred_cost)).get()
                proj, rnorm = zip(*res)
                # the projection
                proj = torch.stack(proj, dim=0).to(device)
                # the residuals
                rnorm = torch.tensor(rnorm).to(device)
            # normalize
            proj = proj / proj.norm(dim=1, keepdim=True)
            # limit the max_iter for clarabel
            if self.solver == "clarabel":
                # already get inner projection
                vec = proj
            # manually push projection inside for scipy.nnls
            else:
                # get excact projection
                # push vector inside by a convex combination of projection and average
                vec = (1 - self.inner_ratio) * proj + self.inner_ratio * avg
                # keep excact projection if already in the cone
                vec[rnorm < 1e-7] = proj[rnorm < 1e-7]
        # hueristic projection
        else:
            # normalize prediction (avoid the unbalanced weights)
            pred_norm = pred_cost / pred_cost.norm(dim=1, keepdim=True)
            # a convex combination of prediction and average
            vec = (1 - self.inner_ratio) * pred_norm + self.inner_ratio * avg
        return vec.detach()

    def _getAvg(self, tight_ctrs):
        """
        A method to get average of binding constraints
        """
        # normalize
        tight_ctrs = tight_ctrs / (tight_ctrs.norm(dim=2, keepdim=True) + 1e-8)
        # average for each instance
        avg = tight_ctrs.mean(dim=1).detach()
        return avg

    @staticmethod
    def _solveClarabel(cp, ctr, max_iter):
        """
        A static method to solve quadratic programming with Clarabel
        """
        # remove zero-padding from binding constraints
        ctr = ctr[np.abs(ctr).sum(axis=1) > 1e-7]
        # decision varibles λ
        λ = cvx.Variable(len(ctr), name="λ", nonneg=True)
        # onjective function ||cp - λ ctr||^2
        objective = cvx.Minimize(cvx.sum_squares(cp - λ @ ctr))
        # ceate an optimization problem model
        problem = cvx.Problem(objective)
        # solve with Clarabel with limited iteration for the suboptimal
        problem.solve(solver=cvx.CLARABEL, max_iter=max_iter)
        # get inner projection (which is inside the cone)
        p = λ.value @ ctr
        # compute residuals as the Euclidean distance between prediction and projection
        rnorm = problem.value
        return torch.tensor(p, dtype=torch.float32), rnorm

    @staticmethod
    def _solveNNLS(cp, ctr, max_iter):
        """
        A static method to solve quadratic programming with scipy
        """
        # WARNING: set max_iter will lead infeasibility and the solution will not be available
        # Now, the max_iter is set to unlimited, but the project will be push inside later
        # by a convex combination of exact projection and average binding constraints
        max_iter = None
        # drop pads
        ctr = ctr[np.abs(ctr).sum(axis=1) > 1e-7]
        # solve as non-negaitive least square (using the active set method)
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html
        λ, rnorm = nnls(ctr.T, cp, maxiter=max_iter)
        # obtain the closest projection on the surface of the cone
        p = λ @ ctr
        return torch.tensor(p, dtype=torch.float32), rnorm
