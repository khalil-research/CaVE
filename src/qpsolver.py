#!/usr/bin/env python
"""
Batched first-order QP solver for cone projection
"""

from __future__ import annotations

import torch


def project_apgd(
    tight_ctrs: torch.Tensor,
    signed_cost: torch.Tensor,
    tol_grad: float | None = 1e-4,
    max_iters: int | None = None,
    check_frequency: int = 200,
    power_iter: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A function to batch-project onto the binding-constraint cone via Nesterov APGD
    """
    # iteration cap
    cap = max_iters if max_iters is not None else 10_000
    # pre-transpose
    A = tight_ctrs.contiguous()
    AT = A.transpose(1, 2).contiguous()
    # per-instance step size
    step_size = _power_iter_step_size(A, AT, power_iter)
    # Nesterov momentum schedule
    ks = torch.arange(cap, device=A.device, dtype=A.dtype)
    momenta = (ks - 2.0) / (ks + 1.0)
    momenta[0] = 0.0
    # cold-start state
    B, m, _ = A.shape
    x_curr = torch.zeros(B, m, device=A.device, dtype=A.dtype)
    x_prev = x_curr.clone()
    # chunked compiled loop with outer convergence check
    K = check_frequency
    for k_start in range(0, cap, K):
        K_actual = min(K, cap - k_start)
        # compiled chunk
        x_curr, x_prev = _apgd_chunk(
            A, AT, signed_cost, step_size, x_curr, x_prev, momenta[k_start:k_start + K_actual],
        )
        # clone breaks cudagraphs output-reuse aliasing
        x_curr = x_curr.clone()
        x_prev = x_prev.clone()
        # tol_grad=None: skip convergence check
        if tol_grad is None:
            continue
        # projected-gradient L-inf norm
        res = torch.bmm(AT, x_curr.unsqueeze(-1)).squeeze(-1) - signed_cost
        grad = torch.bmm(A, res.unsqueeze(-1)).squeeze(-1)
        active = x_curr > 0
        proj_grad = torch.where(active, grad.abs(), torch.clamp(-grad, min=0))
        # convergence check
        if proj_grad.max().item() <= tol_grad:
            break
    # final projection
    proj = torch.bmm(AT, x_curr.unsqueeze(-1)).squeeze(-1)
    # squared residual
    rnorm = (signed_cost - proj).pow(2).sum(dim=1)
    return proj, rnorm


def _power_iter_step_size(
    A: torch.Tensor, AT: torch.Tensor, n_iter: int,
) -> torch.Tensor:
    """A function to estimate per-instance 1/lambda_max(A A^T) via power iteration"""
    B, m, _ = A.shape
    # random init + unit normalize
    v = torch.randn(B, m, device=A.device, dtype=A.dtype)
    v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-8)
    # power iteration: v <- (A A^T) v
    for _ in range(n_iter):
        u = torch.bmm(AT, v.unsqueeze(-1)).squeeze(-1)
        v = torch.bmm(A, u.unsqueeze(-1)).squeeze(-1)
        # re-normalize
        v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-8)
    # Rayleigh quotient lambda = ||A^T v||^2
    u = torch.bmm(AT, v.unsqueeze(-1)).squeeze(-1)
    lam = (u * u).sum(dim=1)
    # invert for step size
    return (1.0 / lam.clamp(min=1e-8)).view(-1, 1)


@torch.compile(mode="reduce-overhead", dynamic=False)
def _apgd_chunk(
    A: torch.Tensor,
    AT: torch.Tensor,
    y: torch.Tensor,
    step_size: torch.Tensor,
    x_curr: torch.Tensor,
    x_prev: torch.Tensor,
    momenta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A fused chunk of Nesterov-accelerated PGD iterations"""
    n_iter = momenta.shape[0]
    for i in range(n_iter):
        # Nesterov extrapolation
        momentum = momenta[i]
        y_k = x_curr + momentum * (x_curr - x_prev)
        # residual A^T y_k - y
        res = torch.bmm(AT, y_k.unsqueeze(-1)).squeeze(-1) - y
        # gradient A (A^T y_k - y)
        grad = torch.bmm(A, res.unsqueeze(-1)).squeeze(-1)
        # save previous iterate
        x_prev = x_curr
        # projected gradient step onto x >= 0
        x_curr = torch.clamp(y_k - step_size * grad, min=0.0)
    return x_curr, x_prev
