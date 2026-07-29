"""Inverse-variance fusion of the top-k active experts, then aggregate-and-project."""

from __future__ import annotations

import torch
from torch import nn


_EPS = 1e-6


class InverseVarianceFusion(nn.Module):
    """Closed-form combination of the top-k experts' ``(mu, sigma^2)``.

    ::

        mu_star       = sum(mu_e / sigma_e^2) / sum(1 / sigma_e^2)
        sigma_star_sq = 1 / sum(1 / sigma_e^2)

    Experts that report high variance are down-weighted automatically, so an
    expert admitting it doesn't know barely moves the final estimate.
    Deliberately ignores the router's gate weight (see ``router.py``): here,
    each expert's own reported uncertainty decides how much it counts, not
    how confident the router was in selecting it.
    """

    def forward(self, mus: torch.Tensor, log_sigma_sqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse ``mus (N, top_k, hidden_dim)`` and ``log_sigma_sqs (N, top_k, 1)``.

        Returns ``(mu_star, log_sigma_star_sq)``, shapes ``(N, hidden_dim)``
        and ``(N, 1)``.
        """
        inv_var = 1.0 / log_sigma_sqs.exp().clamp_min(_EPS)  # (N, top_k, 1)
        sum_inv_var = inv_var.sum(dim=1).clamp_min(_EPS)  # (N, 1)
        mu_star = (inv_var * mus).sum(dim=1) / sum_inv_var
        sigma_star_sq = 1.0 / sum_inv_var
        log_sigma_star_sq = sigma_star_sq.clamp_min(_EPS).log()
        return mu_star, log_sigma_star_sq


class AggregateProject(nn.Module):
    """Projects ``concat(mu_star, log_sigma_star_sq)`` into the evidence token."""

    def __init__(self, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, mu_star: torch.Tensor, log_sigma_star_sq: torch.Tensor) -> torch.Tensor:
        """Project ``concat(mu_star, log_sigma_star_sq)`` into the evidence token."""
        return self.proj(torch.cat([mu_star, log_sigma_star_sq], dim=-1))
