#!/usr/bin/env python
# coding: utf-8
"""
optDataset class to obtain tight constraints
"""

import time

from gurobipy import GRB
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from pyepo.model.opt import optModel


class optDatasetConstrs(Dataset):
    """
    This class is Torch Dataset for optimization problems with active constraints.

    Attributes:
        model (optModel): Optimization models
        feats (np.ndarray): Data features
        costs (np.ndarray): Cost vectors
        sols (np.ndarray): Optimal solutions
        ctrs (list(np.ndarray)): active constraints
    """

    def __init__(self, model, feats, costs=None, sols=None):
        """
        A method to create a optDataset from optModel

        Args:
            model (optModel): an instance of optModel
            feats (np.ndarray): data features
            costs (np.ndarray): costs of objective function
            sols (np.ndarray): optimal solutions
        """
        if not isinstance(model, optModel):
            raise TypeError("arg model is not an optModel")
        if (costs is None) and (sols is None):
            raise ValueError("At least one of 'costs' or 'sols' must be provided.")
        self.model = model
        # data
        self.feats = feats
        # find optimal solutions and tight constraints
        if sols is None:
            self.costs = costs
            self.sols, self.ctrs = self._getSols()
        # get tight constraints
        else:
            self.costs = None
            self.sols = sols
            self.ctrs = self._getCtrs()

    def _getSols(self):
        """
        A method to get optimal solutions for all cost vectors
        """
        sols, ctrs = [], []
        print("Optimizing for optDataset...")
        time.sleep(1)
        for c in tqdm(self.costs):
            try:
                # solve
                sol = self._solve(c)
                # get constrs
                constrs = self._getBindingConstrs(self.model._model)
            except:
                raise ValueError(
                    "For optModel, the method 'solve' should return solution vector and objective value."
                )
            sols.append(sol)
            ctrs.append(np.array(constrs))
        return np.array(sols), ctrs

    def _getCtrs(self):
        """
        A method to get the binding constraints from given solution
        """
        ctrs = []
        print("Obtaining constraints for optDataset...")
        time.sleep(1)
        for sol in tqdm(self.sols):
            # give sol
            model = self._assignSol(sol)
            # get constrs
            constrs = self._getBindingConstrs(model)
            ctrs.append(np.array(constrs))
        return ctrs

    def _solve(self, cost):
        """
        A method to solve optimization problem to get an optimal solution with given cost

        Args:
            cost (np.ndarray): cost of objective function

        Returns:
            tuple: optimal solution (np.ndarray) and objective value (float)
        """
        self.model.setObj(cost)
        sol, _ = self.model.solve()
        return sol

    def _assignSol(self, sol):
        """
        A method to fix model with given solution

        Args:
            sols (np.ndarray): Optimal solutions

        Returns:
            model (optModel): Optimization models
        """
        # copy model
        model = self.model.copy()
        # fix value
        for i, k in enumerate(self.model.x):
            model.x[k].lb = sol[i]
            model.x[k].ub = sol[i]
        # set 0 obj
        model._model.setObjective(0)
        # solve
        model._model.optimize()
        return model._model


    def _getBindingConstrs(self, model):
        """
        A method to get tight constraints with current solution

        Args:
            model (optModel): Optimization models

        Returns:
            np.ndarray: normal vector of constraints
        """
        xs = model.getVars()
        constrs = []
        # iterate all constraints
        for constr in model.getConstrs():
            # check tight constraints
            if abs(constr.Slack) < 1e-5:
                t_constr = []
                # get coefficients
                for x in xs:
                    t_constr.append(model.getCoeff(constr, x))
                # get coefficients in standard form
                if constr.sense == GRB.LESS_EQUAL:
                    # <=
                    constrs.append(t_constr)
                elif constr.sense == GRB.GREATER_EQUAL:
                    # >=
                    constrs.append([- coef for coef in t_constr])
                elif constr.sense == GRB.EQUAL:
                    # ==
                    constrs.append(t_constr)
                    constrs.append([- coef for coef in t_constr])
                else:
                    # invalid sense
                    raise ValueError("Invalid constraint sense.")
        # iterate all variables
        for i, x in enumerate(xs):
            t_constr = [0] * len(xs)
            # add bounds as cosnrtaints
            if x.x <= 1e-5:
                # x_i >= 0
                t_constr[i] = - 1
                constrs.append(t_constr)
            elif x.x >= 1 - 1e-5:
                # x_i <= 1
                t_constr[i] = 1
                constrs.append(t_constr)
        return constrs

    def __len__(self):
        """
        A method to get data size

        Returns:
            int: the number of optimization problems
        """
        return len(self.feats)

    def __getitem__(self, index):
        """
        A method to retrieve data

        Args:
            index (int): data index

        Returns:
            tuple: data features (torch.tensor),
                   costs (torch.tensor),
                   optimal solutions (torch.tensor),
                   objective values (torch.tensor)
        """
        if self.costs is None:
            return (
                torch.FloatTensor(self.feats[index]),
                torch.FloatTensor(self.sols[index]),
                torch.FloatTensor(self.ctrs[index])
            )
        else:
            return (
                torch.FloatTensor(self.feats[index]),
                torch.FloatTensor(self.costs[index]),
                torch.FloatTensor(self.sols[index]),
                torch.FloatTensor(self.ctrs[index])
            )
