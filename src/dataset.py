#!/usr/bin/env python
# coding: utf-8
"""
optDataset class to obtain tight constraints
"""

import time

from gurobipy import GRB
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from pyepo.model.opt import optModel


class optDatasetConstrs(Dataset):
    """
    This class is Torch Dataset for optimization problems with binding constraints.

    Attributes:
        model (optModel): Optimization models
        feats (np.ndarray): Data features
        costs (np.ndarray): Cost vectors
        sols (np.ndarray): Optimal solutions
        ctrs (list(np.ndarray)): active constraints
    """

    def __init__(self, model, feats, costs=None, sols=None, skip_infeas=False):
        """
        A method to create a optDataset from optModel

        Args:
            model (optModel): an instance of optModel
            feats (np.ndarray): data features
            costs (np.ndarray): costs of objective function
            sols (np.ndarray): optimal solutions
            skip_infeas (bool): if True, skip infeasible data points
        """
        if not isinstance(model, optModel):
            raise TypeError("arg model is not an optModel")
        if (costs is None) and (sols is None):
            raise ValueError("At least one of 'costs' or 'sols' must be provided.")
        self.model = model
        # drop infeasibe or get error
        self.skip_infeas = skip_infeas
        # data
        self.feats = feats
        # find optimal solutions and tight constraints
        if sols is None:
            self.costs = costs
            self.sols, self.ctrs = self._getSols()
        # get tight constraints with given optimal solution
        else:
            self.costs = None
            self.sols = sols
            self.ctrs = self._getCtrs()

    def _getSols(self):
        """
        A method to get optimal solutions for all cost vectors
        """
        sols, ctrs, valid_ind = [], [], []
        print("Optimizing for optDataset...")
        time.sleep(1)
        tbar = tqdm(self.costs)
        for i, c in enumerate(tbar):
            try:
                # solve
                sol, model = self._solve(c)
                # get binding constrs
                constrs = self._getBindingConstrs(model)
            except AttributeError as e:
                # infeasibe
                if self.skip_infeas:
                    # skip this data point
                    tbar.write("No feasible solution! Drop instance {}.".format(i))
                    continue
                else:
                    # raise the exception
                    raise ValueError("No feasible solution!")
            sols.append(sol)
            ctrs.append(np.array(constrs))
            valid_ind.append(i)
        # update feats and costs to keep only valid entries
        self.feats = self.feats[valid_ind]
        self.costs = self.costs[valid_ind]
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
        # copy model
        model = self.model.copy()
        # set obj
        model.setObj(cost)
        # optimize
        sol, _ = model.solve()
        return sol, model

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
        # fix all value
        for i, var_x in enumerate(model._model.getVars()):
            var_x.lb = sol[i]
            var_x.ub = sol[i]
        # set 0 obj
        model._model.setObjective(0)
        # solve
        model._model.optimize()
        return model


    def _getBindingConstrs(self, model):
        """
        A method to get tight constraints with current solution

        Args:
            model (optModel): Optimization models

        Returns:
            np.ndarray: normal vector of constraints
        """
        xs = model._model.getVars()
        constrs = []
        # if there is lazy constraints
        if hasattr(model, "lazy_constrs"):
            # add lazy constrs to model
            for constr in model.lazy_constrs:
                model._model.addConstr(constr)
        model._model.update()
        # solve
        model.solve()
        # iterate all constraints
        for constr in model._model.getConstrs():
            # check binding constraints A x == b
            if abs(constr.Slack) < 1e-5:
                t_constr = []
                # get coefficients
                for x in xs:
                    t_constr.append(model._model.getCoeff(constr, x))
                # get coefficients with correct direction
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
        # iterate all variables to check bounds
        for i, x in enumerate(xs):
            t_constr = [0] * len(xs)
            # add tight bounds as cosnrtaints
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


def collate_fn(batch):
    """
    A custom collate function for PyTorch DataLoader.
    """
    # seperate batch data
    x, c, w, t_ctrs = zip(*batch)
    # stack lists of x, c, and w into new batch tensors
    x = torch.stack(x, dim=0)
    c = torch.stack(c, dim=0)
    w = torch.stack(w, dim=0)
    # pad t_ctrs with 0 to make all sequences have the same length.
    # the number of binding constraints are different.
    ctrs_padded = pad_sequence(t_ctrs, batch_first=True, padding_value=0)
    return x, c, w, ctrs_padded