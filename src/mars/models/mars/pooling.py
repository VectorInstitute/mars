"""Attention pooling over spatial tokens."""

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
