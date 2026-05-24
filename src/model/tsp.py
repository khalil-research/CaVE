#!/usr/bin/env python
"""
Traveling salesman problem with binding-constraint tracking for CaVE
"""

from __future__ import annotations

from itertools import combinations

import gurobipy as gp
from gurobipy import GRB

from pyepo.model.grb.tsp import tspDFJModel as _tspDFJModel
from pyepo.model.utils import _EDGE_ACTIVE_TOL, unionFind


class tspDFJModel(_tspDFJModel):
    """
    DFJ TSP model extended with per-solve tracking of lazily-added subtour
    elimination constraints. The active subtour cuts at the optimum are
    needed by CaVE to assemble the cone of binding constraints.
    """

    def _getModel(self) -> tuple:
        m, x = super()._getModel()
        m._lazy_constrs = []
        return m, x

    @staticmethod
    def _subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            xvals = model.cbGetSolution(model._x)
            selected = gp.tuplelist(
                (i, j) for i, j in model._x if xvals[i, j] > _EDGE_ACTIVE_TOL
            )
            uf = unionFind(model._n)
            for i, j in selected:
                if not uf.union(i, j):
                    cycle = [k for k in range(model._n) if uf.find(k) == uf.find(i)]
                    if len(cycle) < model._n:
                        constr = (
                            gp.quicksum(model._x[i, j] for i, j in combinations(cycle, 2))
                            <= len(cycle) - 1
                        )
                        model.cbLazy(constr)
                        model._lazy_constrs.append(constr)
                    break
