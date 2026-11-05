"""The Mamba residual block DeMamba uses, after the mamba.py port by alxndrTL.

The selective scan runs step by step (the released code's sequential mode, equal to
its parallel scan). The second scan reads the input with its channels reversed, as
released; both scans run forward in time.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square normalisation with a learned scale."""

    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale x by the inverse RMS over its last dimension."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SelectiveScan(nn.Module):
    """Input-dependent step size, B and C, and the state-space recurrence."""

    def __init__(self, inner: int, d_state: int, dt_rank: int) -> None:
        super().__init__()
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.x_proj = nn.Linear(inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, inner)
        std = dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(torch.rand(inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(inner, 1)))
        self.D = nn.Parameter(torch.ones(inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scan (N, L, E): h_t = exp(dt A) h_(t-1) + dt B x_t, y_t = C h_t + D x_t."""
        delta, b, c = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        decay = torch.exp(delta.unsqueeze(-1) * -torch.exp(self.A_log))
        inject = delta.unsqueeze(-1) * b.unsqueeze(2) * x.unsqueeze(-1)
        h = x.new_zeros(x.shape[0], x.shape[2], self.d_state)
        outputs = []
        for t in range(x.shape[1]):
            h = decay[:, t] * h + inject[:, t]
            outputs.append((h @ c[:, t].unsqueeze(-1)).squeeze(-1))
        return torch.stack(outputs, dim=1) + self.D * x


class MambaBlock(nn.Module):
    """Gated Mamba mixer: depthwise causal convolution, two selective scans, gate."""

    def __init__(self, width: int, d_state: int, expand: int, d_conv: int) -> None:
        super().__init__()
        inner = expand * width
        dt_rank = math.ceil(width / 16)
        self.in_proj = nn.Linear(width, 2 * inner, bias=False)
        self.conv1d = nn.Conv1d(inner, inner, d_conv, groups=inner, padding=d_conv - 1)
        self.scan = SelectiveScan(inner, d_state, dt_rank)
        self.scan_b = SelectiveScan(inner, d_state, dt_rank)
        self.out_proj = nn.Linear(inner, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix (N, L, D) tokens; each output depends only on inputs up to its step."""
        length = x.shape[1]
        x, gate = self.in_proj(x).chunk(2, dim=-1)
        x = F.silu(self.conv1d(x.transpose(1, 2))[:, :, :length].transpose(1, 2))
        y = self.scan(x) + self.scan_b(x.flip(-1)).flip(-1)
        return self.out_proj(y * F.silu(gate))


class ResidualMamba(nn.Module):
    """x + DropPath(Mamba(RMSNorm(x)))."""

    def __init__(self, width: int, d_state: int, expand: int, d_conv: int, drop_path: float) -> None:
        super().__init__()
        self.norm = RMSNorm(width)
        self.mixer = MambaBlock(width, d_state, expand, d_conv)
        self.drop_path = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block; in training, drop the branch per sequence."""
        branch = self.mixer(self.norm(x))
        if self.training and self.drop_path > 0:
            keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= self.drop_path
            branch = branch * keep.to(branch.dtype) / (1 - self.drop_path)
        return x + branch
