"""Effort's SVD-split linear layer and its two constraints.

A pretrained weight W = U S V^T is split into its top singular directions, kept
frozen, and the remaining ones, which train. The constraints keep the singular
vectors orthogonal and the weight's Frobenius norm at its pretrained value.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SVDResidualLinear(nn.Module):
    """Linear layer with weight W_main (frozen) + U_res diag(S_res) V_res (trained).

    Parameters
    ----------
    weight, bias : torch.Tensor
        The pretrained layer; the bias stays frozen.
    residual_rank : int
        How many of the smallest singular directions train.
    """

    weight_main: torch.Tensor
    bias: torch.Tensor
    U_r: torch.Tensor
    V_r: torch.Tensor
    norm_original: torch.Tensor

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, residual_rank: int) -> None:
        super().__init__()
        u, s, vh = torch.linalg.svd(weight.detach().float(), full_matrices=False)
        r = len(s) - residual_rank
        # Rebuilt from the pretrained weight at construction, so checkpoints skip them.
        self.register_buffer("weight_main", u[:, :r] @ torch.diag(s[:r]) @ vh[:r], persistent=False)
        self.register_buffer("bias", bias.detach().float().clone(), persistent=False)
        self.register_buffer("U_r", u[:, :r].clone(), persistent=False)
        self.register_buffer("V_r", vh[:r].clone(), persistent=False)
        self.register_buffer("norm_original", torch.linalg.norm(weight.detach().float()), persistent=False)
        self.U_residual = nn.Parameter(u[:, r:].clone())
        self.S_residual = nn.Parameter(s[r:].clone())
        self.V_residual = nn.Parameter(vh[r:].clone())

    def weight(self) -> torch.Tensor:
        """Return the current weight, frozen part plus residual."""
        return self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the layer with the current weight."""
        return F.linear(x, self.weight(), self.bias)

    def orthogonal_loss(self) -> torch.Tensor:
        """0.5 ||U U^T - I||_F + 0.5 ||V V^T - I||_F over all singular vectors."""
        u = torch.cat([self.U_r, self.U_residual], dim=1)
        v = torch.cat([self.V_r, self.V_residual], dim=0)
        eye_u = torch.eye(len(u), device=u.device)
        eye_v = torch.eye(len(v), device=v.device)
        return 0.5 * torch.linalg.norm(u @ u.T - eye_u) + 0.5 * torch.linalg.norm(v @ v.T - eye_v)

    def singular_value_loss(self) -> torch.Tensor:
        """| ||W||_F^2 - ||W_pretrained||_F^2 |."""
        return (torch.linalg.norm(self.weight()) ** 2 - self.norm_original**2).abs()
