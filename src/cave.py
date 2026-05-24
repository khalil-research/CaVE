#!/usr/bin/env python
"""
Cone-aligned vector estimation (CaVE) loss for binary linear programs
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.optimize import nnls
from torch.nn import functional as F

from pyepo import EPO
from pyepo.func.abcmodule import optModule

try:
    import cvxpy as cvx
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False

if TYPE_CHECKING:
    from pyepo.func.abcmodule import Reduction
    from pyepo.model.opt import optModel


class abstractConeAlignedCosine(optModule):
    """
    An abstract autograd module for the CaVE family of cone-aligned cosine
    losses. Subclasses project the sense-flipped predicted cost vector onto
    the polyhedral cone spanned by binding-constraint normals at the optimal
    vertex; the loss is ``1 - cos(pred, proj)``.

    Reference: <https://link.springer.com/chapter/10.1007/978-3-031-60599-4_12>
    """

    def __init__(
        self,
        optmodel: optModel,
        processes: int = 1,
        reduction: Reduction = "mean",
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model
            processes: number of processors, 1 for single-core, 0 for all of cores
            reduction: the reduction to apply to the output
        """
        super().__init__(optmodel, processes, solve_ratio=1.0, reduction=reduction)

    def forward(
        self, pred_cost: torch.Tensor, tight_ctrs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            pred_cost: (B, n) predicted cost vectors
            tight_ctrs: (B, max_ctrs, n) zero-padded binding-constraint normals

        Returns:
            torch.Tensor: reduced cone-alignment loss in [0, 2]
        """
        if self.optmodel.modelSense == EPO.MINIMIZE:
            sign = -1.0
        elif self.optmodel.modelSense == EPO.MAXIMIZE:
            sign = 1.0
        else:
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        signed_cost = sign * pred_cost
        proj = self._get_projection(signed_cost, tight_ctrs)
        loss = 1.0 - F.cosine_similarity(signed_cost, proj, dim=1)
        return self._reduce(loss)

    @abstractmethod
    def _get_projection(
        self, signed_cost: torch.Tensor, tight_ctrs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the projection target for each instance in the batch.

        Args:
            signed_cost: (B, n) sense-flipped predicted cost
            tight_ctrs: (B, max_ctrs, n) zero-padded binding-constraint normals

        Returns:
            torch.Tensor: (B, n) projection target, detached from autograd
        """


class exactConeAlignedCosine(abstractConeAlignedCosine):
    """
    An autograd module for the CaVE Exact loss. Solves a full non-negative
    least-squares projection of the sense-flipped predicted cost onto the
    cone spanned by binding-constraint normals at the optimal vertex.

    Reference: <https://link.springer.com/chapter/10.1007/978-3-031-60599-4_12>
    """

    def __init__(
        self,
        optmodel: optModel,
        solver: str = "clarabel",
        processes: int = 1,
        reduction: Reduction = "mean",
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model
            solver: QP solver for the projection, 'clarabel' (cvxpy) or 'nnls' (scipy)
            processes: number of processors, 1 for single-core, 0 for all of cores
            reduction: the reduction to apply to the output
        """
        super().__init__(optmodel, processes, reduction)
        if solver not in ("clarabel", "nnls"):
            raise ValueError(f"Invalid solver: {solver}. Must be 'clarabel' or 'nnls'.")
        if solver == "clarabel" and not _HAS_CVXPY:
            raise ImportError(
                "cvxpy is not installed. Install with `pip install 'cvxpy[clarabel]'` "
                "to use the 'clarabel' solver, or pass solver='nnls'."
            )
        self.solver = solver

    def _get_projection(
        self, signed_cost: torch.Tensor, tight_ctrs: torch.Tensor,
    ) -> torch.Tensor:
        proj, _ = _batch_project(
            signed_cost, tight_ctrs, self.solver, max_iter=None,
            processes=self.processes, pool=self.pool,
        )
        return proj / proj.norm(dim=1, keepdim=True).clamp(min=1e-8)


class innerConeAlignedCosine(exactConeAlignedCosine):
    """
    An autograd module for the CaVE+ and CaVE Hybrid losses. Combines an
    inner-truncated cone projection (Clarabel with early termination, or
    NNLS pushed inside via a convex combination with the average normal)
    with an optional cheap heuristic branch.

    With ``solve_ratio == 1`` this is CaVE+; with ``solve_ratio < 1`` it is
    CaVE Hybrid, taking the heuristic branch with probability ``1 - solve_ratio``.

    Reference: <https://link.springer.com/chapter/10.1007/978-3-031-60599-4_12>
    """

    def __init__(
        self,
        optmodel: optModel,
        solver: str = "clarabel",
        max_iter: int = 3,
        solve_ratio: float = 1.0,
        inner_ratio: float = 0.2,
        processes: int = 1,
        reduction: Reduction = "mean",
        seed: int | None = None,
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model
            solver: QP solver for the projection, 'clarabel' (cvxpy) or 'nnls' (scipy)
            max_iter: Clarabel iteration budget for the inner-truncated projection
            solve_ratio: probability per batch of running the QP projection;
                with complementary probability the cheap heuristic branch is taken
            inner_ratio: weight on the average normal in the heuristic and in
                the post-NNLS push-inside step
            processes: number of processors, 1 for single-core, 0 for all of cores
            reduction: the reduction to apply to the output
            seed: seed for the per-batch QP-vs-heuristic branch RNG
        """
        super().__init__(optmodel, solver, processes, reduction)
        if not 0 <= solve_ratio <= 1:
            raise ValueError(
                f"Invalid solve_ratio {solve_ratio}. It should be between 0 and 1."
            )
        if not 0 <= inner_ratio <= 1:
            raise ValueError(
                f"Invalid inner_ratio {inner_ratio}. It should be between 0 and 1."
            )
        self.max_iter = int(max_iter)
        self.solve_ratio = float(solve_ratio)
        self.inner_ratio = float(inner_ratio)
        if seed is not None:
            self._branch_rng = np.random.RandomState(seed)

    def _get_projection(
        self, signed_cost: torch.Tensor, tight_ctrs: torch.Tensor,
    ) -> torch.Tensor:
        avg = _average_ctrs(tight_ctrs)
        # heuristic branch: skip the QP, convex-combine pred with the average normal
        if self._branch_rng.uniform() > self.solve_ratio:
            pred_norm = signed_cost / signed_cost.norm(dim=1, keepdim=True).clamp(min=1e-8)
            return ((1 - self.inner_ratio) * pred_norm + self.inner_ratio * avg).detach()
        # QP branch with truncated iterations
        proj, rnorm = _batch_project(
            signed_cost, tight_ctrs, self.solver, max_iter=self.max_iter,
            processes=self.processes, pool=self.pool,
        )
        proj_norm = proj / proj.norm(dim=1, keepdim=True).clamp(min=1e-8)
        # Clarabel with truncated iterations already lands strictly inside the cone
        if self.solver == "clarabel":
            return proj_norm.detach()
        # NNLS solves to the cone boundary, so nudge inside via the average normal,
        # except for instances already inside (zero residual) which keep the exact projection
        pushed = (1 - self.inner_ratio) * proj_norm + self.inner_ratio * avg
        inside = (rnorm < 1e-7).unsqueeze(1)
        return torch.where(inside, proj_norm, pushed).detach()


def _average_ctrs(tight_ctrs: torch.Tensor) -> torch.Tensor:
    """Per-instance average of unit-normalized binding-constraint normals (zero-padded rows excluded)."""
    norms = tight_ctrs.norm(dim=2, keepdim=True)
    valid = (norms > 1e-7).to(tight_ctrs.dtype)
    unit = tight_ctrs / norms.clamp(min=1e-8) * valid
    n_valid = valid.sum(dim=1).clamp(min=1.0)
    return (unit.sum(dim=1) / n_valid).detach()


def _batch_project(
    signed_cost: torch.Tensor,
    tight_ctrs: torch.Tensor,
    solver: str,
    max_iter: int | None,
    processes: int,
    pool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map per-instance cone projections; returns (proj, rnorm) on signed_cost's device."""
    device, dtype = signed_cost.device, signed_cost.dtype
    cp_np = signed_cost.detach().cpu().numpy()
    ctrs_np = tight_ctrs.detach().cpu().numpy()
    batch = len(cp_np)
    if solver == "clarabel":
        worker = _project_clarabel
    elif solver == "nnls":
        worker = _project_nnls
    else:
        raise ValueError(f"Invalid solver: {solver}. Must be 'clarabel' or 'nnls'.")
    if processes == 1:
        results = [worker(cp_np[i], ctrs_np[i], max_iter) for i in range(batch)]
    else:
        results = pool.amap(worker, list(cp_np), list(ctrs_np), [max_iter] * batch).get()
    proj_np = np.stack([r[0] for r in results])
    rnorm_np = np.asarray([r[1] for r in results], dtype=np.float32)
    proj = torch.as_tensor(proj_np, dtype=dtype, device=device)
    rnorm = torch.as_tensor(rnorm_np, dtype=dtype, device=device)
    return proj, rnorm


def _project_clarabel(
    cp: np.ndarray, ctr: np.ndarray, max_iter: int | None,
) -> tuple[np.ndarray, float]:
    """Project cp onto cone{lam @ ctr : lam >= 0} via Clarabel; returns (projection, residual norm)."""
    ctr = ctr[np.abs(ctr).sum(axis=1) > 1e-7]
    if len(ctr) == 0:
        return cp.astype(np.float32), 0.0
    lam = cvx.Variable(len(ctr), nonneg=True)
    objective = cvx.Minimize(cvx.sum_squares(cp - lam @ ctr))
    problem = cvx.Problem(objective)
    solve_kwargs: dict = {"solver": cvx.CLARABEL}
    if max_iter is not None:
        solve_kwargs["max_iter"] = max_iter
    problem.solve(**solve_kwargs)
    if lam.value is None:
        return cp.astype(np.float32), float("inf")
    p = lam.value @ ctr
    return p.astype(np.float32), float(problem.value)


def _project_nnls(
    cp: np.ndarray, ctr: np.ndarray, max_iter: int | None,
) -> tuple[np.ndarray, float]:
    """Project cp onto cone{lam @ ctr : lam >= 0} via SciPy NNLS; returns (projection, residual norm)."""
    del max_iter
    ctr = ctr[np.abs(ctr).sum(axis=1) > 1e-7]
    if len(ctr) == 0:
        return cp.astype(np.float32), 0.0
    ctr_T = np.asfortranarray(ctr.T)
    lam, rnorm = nnls(ctr_T, cp)
    p = lam @ ctr
    return p.astype(np.float32), float(rnorm)
