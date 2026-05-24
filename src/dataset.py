#!/usr/bin/env python
"""
optDataset with binding-constraint extraction for CaVE
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from gurobipy import GRB
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from pyepo.data.dataset import optDataset
from pyepo.model.opt import optModel

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class optDatasetConstrs(optDataset):
    """
    This class is a Torch Dataset for optimization problems with the normals
    of binding constraints at the optimum.

    Reference: <https://link.springer.com/chapter/10.1007/978-3-031-60599-4_12>

    Attributes:
        model (optModel): Optimization model
        feats (torch.Tensor): Data features
        costs (torch.Tensor): Cost vectors
        sols (torch.Tensor): Optimal solutions
        objs (torch.Tensor): Optimal objective values
        ctrs (list[torch.Tensor]): Per-instance binding-constraint normals
    """

    def __init__(
        self,
        model: optModel,
        feats: np.ndarray | torch.Tensor,
        costs: np.ndarray | torch.Tensor,
        skip_infeas: bool = False,
    ) -> None:
        """
        A method to create an optDatasetConstrs from optModel

        Args:
            model: an instance of optModel
            feats: data features
            costs: costs of objective function
            skip_infeas: if True, drop infeasible instances instead of raising
        """
        if not isinstance(model, optModel):
            raise TypeError("arg model is not an optModel")
        self.model = model
        self.skip_infeas = skip_infeas
        # data
        self.feats = feats
        self.costs = costs
        # find optimal solutions and binding constraints
        sols, objs, ctrs, valid = self._getSols()
        # pre-convert to tensors (on CPU) to avoid repeated numpy→tensor copies
        self.feats = torch.as_tensor(self.feats[valid], dtype=torch.float32)
        self.costs = torch.as_tensor(self.costs[valid], dtype=torch.float32)
        self.sols = torch.as_tensor(sols, dtype=torch.float32)
        self.objs = torch.as_tensor(objs, dtype=torch.float32)
        self.ctrs = [torch.as_tensor(c, dtype=torch.float32) for c in ctrs]

    def _getSols(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[int]]:
        """
        A method to get optimal solutions and binding-constraint normals
        """
        sols: list[np.ndarray] = []
        objs: list[list[float]] = []
        ctrs: list[np.ndarray] = []
        valid: list[int] = []
        logger.info("Optimizing for optDatasetConstrs...")
        for i, c in enumerate(tqdm(self.costs)):
            # fresh per-instance copy keeps the lazy-constraint buffer clean
            model = self.model.copy()
            model.setObj(c)
            sol, obj = model.solve()
            # infeasibility check
            if model._model.Status != GRB.OPTIMAL:
                if self.skip_infeas:
                    logger.warning(
                        "Instance %d non-optimal (Status=%d), skipping.",
                        i, model._model.Status,
                    )
                    continue
                raise ValueError(
                    f"Instance {i} did not solve to optimality "
                    f"(Gurobi Status={model._model.Status})."
                )
            sols.append(np.asarray(sol))
            objs.append([float(obj)])
            ctrs.append(_extract_tight_normals(model, sol))
            valid.append(i)
        return np.stack(sols), np.asarray(objs), ctrs, valid

    def __len__(self) -> int:
        """
        A method to get data size
        """
        return len(self.feats)

    def __getitem__(
        self, index: int,
    ) -> tuple[torch.Tensor, ...]:
        """
        A method to retrieve data

        Returns:
            tuple: features, costs, optimal solution, optimal objective,
            binding-constraint normals
        """
        return (
            self.feats[index],
            self.costs[index],
            self.sols[index],
            self.objs[index],
            self.ctrs[index],
        )


def collate_fn(batch):
    """
    A custom collate function for PyTorch DataLoader that pads binding-constraint matrices
    """
    x, c, w, z, t_ctrs = zip(*batch)
    return (
        torch.stack(x, dim=0),
        torch.stack(c, dim=0),
        torch.stack(w, dim=0),
        torch.stack(z, dim=0),
        pad_sequence(t_ctrs, batch_first=True, padding_value=0),
    )


def _extract_tight_normals(
    model: optModel, sol: np.ndarray, tol: float = 1e-5,
) -> np.ndarray:
    """
    A function to extract normals of binding constraints at sol in canonical <= orientation
    """
    grb = model._model
    cost_vars: list = model._cost_vars
    num_cost = len(cost_vars)
    sol_np = np.asarray(sol, dtype=np.float64)
    rows: list[np.ndarray] = []
    # explicit constraints
    constrs = grb.getConstrs()
    if constrs:
        slacks = np.asarray(grb.getAttr("Slack", constrs))
        senses = grb.getAttr("Sense", constrs)
        tight_mask = np.abs(slacks) < tol
        if tight_mask.any():
            # project the constraint matrix onto cost-variable columns
            cost_col_idx = np.asarray([v.index for v in cost_vars])
            A = grb.getA().tocsr()
            A_cost = A[:, cost_col_idx]
            for row_idx in np.where(tight_mask)[0]:
                row = A_cost[row_idx].toarray().ravel()
                rows.extend(_orient_row(row, senses[row_idx]))
    # lazy constraints: evaluate LHS at the optimum to derive slack
    var_to_cost: dict[str, int] = {v.VarName: k for k, v in enumerate(cost_vars)}
    for tc in getattr(grb, "_lazy_constrs", []):
        coefs, rhs, sense = _temp_constr_to_cost_row(tc, var_to_cost, num_cost)
        if coefs is None:
            continue
        lhs_val = float(coefs @ sol_np)
        if abs(rhs - lhs_val) < tol:
            rows.extend(_orient_row(coefs, sense))
    # binary variable bounds
    for k in range(num_cost):
        # tight at 0: -x_k <= 0
        if sol_np[k] <= tol:
            row = np.zeros(num_cost, dtype=np.float64)
            row[k] = -1.0
            rows.append(row)
        # tight at 1: x_k <= 1
        elif sol_np[k] >= 1 - tol:
            row = np.zeros(num_cost, dtype=np.float64)
            row[k] = 1.0
            rows.append(row)
    # empty fallback
    if not rows:
        return np.zeros((0, num_cost), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _orient_row(row: np.ndarray, sense: str) -> list[np.ndarray]:
    """Return constraint rows in canonical ``<=`` orientation."""
    # <=
    if sense == GRB.LESS_EQUAL:
        return [row]
    # >= negated to <=
    if sense == GRB.GREATER_EQUAL:
        return [-row]
    # == split into <= and >=
    if sense == GRB.EQUAL:
        return [row, -row]
    raise ValueError(f"Invalid constraint sense {sense!r}.")


def _temp_constr_to_cost_row(
    tc, var_to_cost: dict[str, int], num_cost: int,
) -> tuple[np.ndarray | None, float | None, str | None]:
    """
    Parse a Gurobi TempConstr into (coefs, rhs, sense) over the cost-vector dim
    """
    # TempConstr internals
    lhs = getattr(tc, "_lhs", None)
    rhs = getattr(tc, "_rhs", None)
    sense = getattr(tc, "_sense", None)
    # unparseable fallback
    if lhs is None or rhs is None or sense is None:
        return None, None, None
    # project LinExpr terms onto cost-vector dim
    coefs = np.zeros(num_cost, dtype=np.float64)
    for i in range(lhs.size()):
        var = lhs.getVar(i)
        k = var_to_cost.get(var.VarName)
        if k is not None:
            coefs[k] += lhs.getCoeff(i)
    return coefs, float(rhs), sense
