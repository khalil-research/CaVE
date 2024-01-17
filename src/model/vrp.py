#!/usr/bin/env python
# coding: utf-8
"""
Vehicle routing probelm
"""
import copy
from collections import defaultdict

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from src.model.utils import unionFind

from pyepo.model.grb.grbmodel import optGrbModel

# WARNING: for the sake of simplicity and the guarantee of pure Binary variables,
# the case where only one node is accessed is not taken into account. However,
# this problem can be handeled by adding a dummy of depot.

class vrpABModel(optGrbModel):
    """
    This abstract class is optimization model for capacitated vehicle routing probelm

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
        demands (list(int)): List of customer demands
        capacity (int): Vehicle capacity
        num_vehicle (int): Number of vehicle
    """

    def __init__(self, num_nodes, demands, capacity, num_vehicle):
        """
        Args:
            num_nodes (int): number of nodes
            demands (list(int)): customer demands
            capacity (int): vehicle capacity
            num_vehicle (int): number of vehicle
        """
        self.num_nodes = num_nodes
        self.nodes = list(range(num_nodes))
        self.edges = [(i, j) for i in self.nodes
                      for j in self.nodes if i < j]
        self.demands = demands
        self.capacity = capacity
        self.num_vehicle = num_vehicle
        super().__init__()

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (list): cost vector
        """
        obj = gp.quicksum(c[i] * self.x[e] for i, e in enumerate(self.edges))
        self._model.setObjective(obj)

    def solve(self):
        """
        A method to solve model
        """
        # solve
        self._model.optimize()
        sol = np.zeros(len(self.edges), dtype=np.uint8)
        for i, e in enumerate(self.edges):
            if self.x[e].x > 1e-2:
                sol[i] = int(np.round(self.x[e].x))
        return sol, self._model.objVal

    def getTour(self, sol):
        """
        A method to get a tour from solution

        Args:
            sol (list): solution

        Returns:
            list: a VRP tour
        """
        # active edges
        edges = defaultdict(list)
        for i, (u, v) in enumerate(self.edges):
            if sol[i] > 1e-2:
                edges[u].append(v)
                edges[v].append(u)
        # get tour
        route = []
        candidates = edges[0]
        while edges[0]:
            v_curr = 0
            tour = [0]
            v_next = edges[v_curr][0]
            # remove used edges
            edges[v_curr].remove(v_next)
            edges[v_next].remove(v_curr)
            while v_next != 0:
                tour.append(v_next)
                # go to next node
                if not edges[v_next]: # visit single customer
                    v_curr, v_next = v_next, 0
                else:
                    v_curr, v_next = v_next, edges[v_next][0]
                    # remove used edges
                    edges[v_curr].remove(v_next)
                    edges[v_next].remove(v_curr)
            # back to depot
            tour.append(0)
            route.append(tour)
        return route

    def copy(self):
        """
        A method to copy model

        Returns:
            optModel: new copied model
        """
        new_model = type(self)(self.num_nodes, self.demands,
                               self.capacity, self.num_vehicle)
        # copy params
        for attr in dir(self._model.Params):
            if not attr.startswith('_'):
                try:
                    # get value
                    val = self._model.getParamInfo(attr)[2]
                    # set value
                    new_model._model.setParam(attr, val)
                except gp.GurobiError:
                    # ignore non-param
                    pass
        return new_model


class vrpModel(vrpABModel):
    """
    This class is optimization model for capacitated vehicle routing probelm

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
        demands (list(int)): List of customer demands
        capacity (int): Vehicle capacity
        num_vehicle (int): Number of vehicle
    """

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("vrp")
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        x = m.addVars(self.edges, name="x", vtype=GRB.BINARY)
        for i, j in self.edges:
            x[j, i] = x[i, j]
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstr(x.sum(0, "*") <= 2 * self.num_vehicle) # depot degree
        m.addConstrs(x.sum(i, "*") == 2 for i in self.nodes if i != 0)  # 2 degree
        # activate lazy constraints
        m._x = x
        m._q = {i: self.demands[i-1] for i in self.nodes[1:]}
        m._Q = self.capacity
        m._edges = self.edges
        m.Params.lazyConstraints = 1
        return m, x

    def _vrpCallback(self, model, where):
        """
        A method to add k-path lazy constraints for CVRP
        """
        if where == GRB.Callback.MIPSOL:
            # check subcycle with unionfind
            uf = unionFind(self.num_nodes)
            for u, v in model._edges:
                if u == 0 or v == 0:
                    continue
                if model.cbGetSolution(model._x[u, v]) > 1e-2:
                    uf.union(u, v)
            # go through subcycles
            for component in uf.getComponents():
                if len(component) < 3:
                    continue
                # rounded capacity inequalities
                k = int(np.ceil(np.sum([model._q[v] for v in component]) / model._Q))
                # edges with both end-vertex in S
                edges_s = [(u, v) for u in component for v in component if u < v]
                # add k-path cut
                if len(component) >= 3:
                    if (len(edges_s) >= len(component)) or (k > 1):
                        # constraint expression
                        constr = gp.quicksum(model._x[e]
                                             for e in edges_s) <= len(component) - k
                        # add lazy constraints
                        model.cbLazy(constr)
                        # store lazy constraints to find all binding constraints
                        self.lazy_constrs.append(constr)

    def solve(self):
        """
        A method to solve model
        """
        # init lazy constraints
        self.lazy_constrs = []
        # create a callback function with access to method variables
        def vrpCallback(model, where):
            self._vrpCallback(model, where)
        # solve
        self._model.optimize(vrpCallback)
        sol = np.zeros(len(self.edges), dtype=np.uint8)
        for i, e in enumerate(self.edges):
            if self.x[e].x > 1e-2:
                sol[i] = int(np.round(self.x[e].x))
        return sol, self._model.objVal


class vrpModel2(vrpABModel):
    """
    This class is optimization model for capacitated vehicle routing probelm

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
        demands (list(int)): List of customer demands
        capacity (int): Vehicle capacity
        num_vehicle (int): Number of vehicle
    """

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("vrp")
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        x = m.addVars(self.edges, name="x", vtype=GRB.BINARY)
        for i, j in self.edges:
            x[j, i] = x[i, j]
        u = m.addVars(self.nodes, ub=sefl.capacity, vtype=GRB.CONTINUOUS)
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstr(x.sum(0, "*") <= 2 * self.num_vehicle) # depot degree
        m.addConstrs(x.sum(i, "*") == 2 for i in self.nodes if i != 0)  # 2 degree
        m.addConstrs((u[i] - u[j] + self.capacity * x[i, j] <= self.capacity - demands[j-1])
                     for i, j in self.edges if i != 0 and j != 0) # capacity
        m.addConstrs(demands[i-1] <= self.u[i] for i in self.nodes if i != 0)
