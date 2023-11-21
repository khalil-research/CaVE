#!/usr/bin/env python
# coding: utf-8
"""
Traveling salesman probelm
"""

from collections import defaultdict
from itertools import combinations

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from pyepo.model.grb.tsp import tspABModel

class unionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i
            return True
        return False

class tspDFJModel(tspABModel):
    """
    This class is optimization model for traveling salesman problem based on Danzig–Fulkerson–Johnson (DFJ) formulation and
    constraint generation.

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("tsp")
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        x = m.addVars(self.edges, name="x", vtype=GRB.BINARY)
        for i, j in self.edges:
            x[j, i] = x[i, j]
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstrs(x.sum(i, "*") == 2 for i in self.nodes)  # 2 degree
        # activate lazy constraints
        m._x = x
        m._n = len(self.nodes)
        m.Params.lazyConstraints = 1
        return m, x

    def _subtourelim(self, model, where):
        """
        A static method to add lazy constraints for subtour elimination
        """
        if where == GRB.Callback.MIPSOL:
            # selected edges
            xvals = model.cbGetSolution(model._x)
            selected = gp.tuplelist(
                (i, j) for i, j in model._x.keys() if xvals[i, j] > 1e-2)
            # check subcycle with unionfind
            uf = unionFind(model._n)
            for i, j in selected:
                if not uf.union(i, j):
                    # find subcycle
                    cycle = [k for k in range(model._n) if uf.find(k) == uf.find(i)]
                    if len(cycle) < model._n:
                        constr = gp.quicksum(model._x[i, j]
                                     for i, j in combinations(cycle, 2)) <= len(cycle) - 1
                        model.cbLazy(constr)
                        self.lazy_constrs.append(constr)
                    break

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (list): cost vector
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector cannot match vars.")
        obj = gp.quicksum(c[i] * self.x[k] for i, k in enumerate(self.edges))
        self._model.setObjective(obj)

    def solve(self):
        """
        A method to solve model
        """
        # init lazy constraints
        self.lazy_constrs = []
        # create a callback function with access to method variables
        def subtourelim(model, where):
            self._subtourelim(model, where)
        # solve
        self._model.update()
        self._model.optimize(subtourelim)
        # get solution
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for i, e in enumerate(self.edges):
            if self.x[e].x > 1e-2:
                sol[i] = 1
        return sol, self._model.objVal

    def addConstr(self, coefs, rhs):
        """
        A method to add new constraint

        Args:
            coefs (ndarray): coeffcients of new constraint
            rhs (float): right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        if len(coefs) != self.num_cost:
            raise ValueError("Size of coef vector cannot cost.")
        # copy
        new_model = self.copy()
        # add constraint
        new_model._model.addConstr(
            gp.quicksum(coefs[i] * new_model.x[k]
                        for i, k in enumerate(new_model.edges)) <= rhs)
        return new_model
