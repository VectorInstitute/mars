"""Spatial pooling for the classifier and AD-loss attention construction."""

from __future__ import annotations

import torch
from torch import nn


class AttentionPool(nn.Module):
    """Learned spatial reduction over patch tokens (attention MIL pooling)."""

    def __init__(self, d_model: int, hidden: int = 128) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool ``(B, N, d_model)`` to ``(B, d_model)`` with weights ``(B, N)``."""
        if x.dim() != 3:
            raise ValueError(f"expected (B, N, d_model), got {tuple(x.shape)}")
        weights = self.score(x).softmax(dim=1)
        return (weights * x).sum(dim=1), weights.squeeze(-1)


def build_attention(heads: torch.Tensor, batch: int, n_tokens: int) -> torch.Tensor:
    """Reshape block-1 head outputs to ``A`` of shape ``(B, n_h, t_s, d_s)``."""
    if heads.shape[0] != batch * n_tokens:
        raise ValueError(f"{heads.shape[0]} rows != {batch} x {n_tokens}")
    return heads.reshape(batch, n_tokens, heads.shape[1], heads.shape[2]).permute(0, 2, 1, 3)


def pooled_features(attention: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """Contract attention ``A`` with encoder tokens ``xi`` to pooled ``P``.

    ``A`` is ``(B, n_h, t_s, d_s)``, ``xi`` is ``(B, n_f, t_s, d_s)``,
    and ``P`` is ``(B, n_h, n_f)``.
    """
    batch, heads, tokens, dim = attention.shape
    if xi.shape[0] != batch or xi.shape[2] != tokens or xi.shape[3] != dim:
        raise ValueError(f"A {tuple(attention.shape)} incompatible with xi {tuple(xi.shape)}")
    flat_a = attention.reshape(batch, heads, tokens * dim)
    flat_xi = xi.reshape(batch, xi.shape[1], tokens * dim).to(flat_a.dtype)
    return torch.bmm(flat_a, flat_xi.transpose(1, 2))


def heatmaps(attention: torch.Tensor, grid: int = 27) -> torch.Tensor:
    """Reduce ``A`` to per-position magnitude on a ``grid x grid`` layout."""
    batch, heads, tokens, _ = attention.shape
    if tokens != grid * grid:
        raise ValueError(f"{tokens} tokens do not form a {grid}x{grid} grid")
    return attention.norm(dim=-1).reshape(batch, heads, grid, grid)
