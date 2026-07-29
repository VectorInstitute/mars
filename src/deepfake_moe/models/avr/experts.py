"""Forensic Expert modules — each predicts a per-token Gaussian, not a point value."""

from __future__ import annotations

import torch
from torch import nn


class ForensicExpert(nn.Module):
    """One expert: fused per-token representation -> ``(mu, log_sigma_sq)``.

    A small 2-layer MLP rather than another attention block — the input
    already went through ``imputation.py``'s cross-attention, so there's no
    remaining token-mixing left to do here.

    ``log_sigma_sq`` is a single scalar per token, not one value per hidden
    dimension — nothing here needs per-dimension variance, and a scalar
    keeps each expert head's size down. Predicting the log rather than
    ``sigma_sq`` directly is standard for heteroscedastic regression: it
    keeps variance positive by construction and avoids gradient blowup as
    ``sigma_sq`` approaches zero.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.mu_head = nn.Linear(hidden_dim, hidden_dim)
        self.log_sigma_sq_head = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, log_sigma_sq)`` for ``tokens`` ``(N, hidden_dim)``.

        ``mu`` is ``(N, hidden_dim)``; ``log_sigma_sq`` is ``(N, 1)``.
        """
        hidden = self.trunk(tokens)
        return self.mu_head(hidden), self.log_sigma_sq_head(hidden)
