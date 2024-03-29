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
from tqdm import tqdm

from pyepo.model.opt import optModel
from pyepo.data.dataset import optDataset


class optDatasetConstrs(optDataset):
    """
    This class is Torch Dataset for optimization problems with binding constraints.

    Attributes:
        model (optModel): Optimization models
        feats (np.ndarray): Data features
        costs (np.ndarray): Cost vectors
        sols (np.ndarray): Optimal solutions
        ctrs (list(np.ndarray)): active constraints
    """
    def __init__(self, model, feats, costs, skip_infeas=False):
        """
        A method to create a optDataset from optModel

        Args:
            model (optModel): an instance of optModel
            feats (np.ndarray): data features
            costs (np.ndarray): costs of objective function
            skip_infeas (bool): if True, skip infeasible data points
        """
        if not isinstance(model, optModel):
            raise TypeError("arg model is not an optModel")
        self.model = model
        # drop infeasibe or get error
        self.skip_infeas = skip_infeas
        # data
        self.feats = feats
        self.costs = costs
        self.sols, self.objs, self.ctrs = self._getSols()

    def _getSols(self):
        """
        A method to get optimal solutions for all cost vectors
        """
        sols, objs, ctrs, valid_ind = [], [], [], []
        print("Optimizing for optDataset...")
        time.sleep(1)
        tbar = tqdm(self.costs)
        for i, c in enumerate(tbar):
            try:
                # solve
                sol, obj, model = self._solve(c)
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
            objs.append([obj])
            ctrs.append(np.array(constrs))
            valid_ind.append(i)
        # update feats and costs to keep only valid entries
        self.feats = self.feats[valid_ind]
        self.costs = self.costs[valid_ind]
        return np.array(sols), np.array(objs), ctrs

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
        sol, obj = model.solve()
        return sol, obj, model

    def _getBindingConstrs(self, model):
        """
        A method to get tight constraints with current solution

        Args:
            model (optModel): optimization models

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
            # fix the variables to the optimal
            for var in model._model.getVars():
                var.start = int(var.x)
            # update model
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
        return (
                torch.FloatTensor(self.feats[index]),
                torch.FloatTensor(self.costs[index]),
                torch.FloatTensor(self.sols[index]),
                torch.FloatTensor(self.objs[index]),
                torch.FloatTensor(self.ctrs[index])
            )


def collate_fn(batch):
    """
    A custom collate function for PyTorch DataLoader.
    """
    # seperate batch data
    x, c, w, z, t_ctrs = zip(*batch)
    # stack lists of x, c, and w into new batch tensors
    x = torch.stack(x, dim=0)
    c = torch.stack(c, dim=0)
    w = torch.stack(w, dim=0)
    z = torch.stack(z, dim=0)
    # pad t_ctrs with 0 to make all sequences have the same length.
    # the number of binding constraints are different.
    ctrs_padded = pad_sequence(t_ctrs, batch_first=True, padding_value=0)
    return x, c, w, z, ctrs_padded
