"""Sine-cosine temporal positional encoding."""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_encoding(length: int, dim: int) -> torch.Tensor:
    """Return a ``(length, dim)`` positional table."""
    if dim % 2:
        raise ValueError(f"positional dim must be even, got {dim}")
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    index = torch.arange(0, dim, 2, dtype=torch.float32)
    frequency = torch.exp(-math.log(10000.0) * index / dim)
    table = torch.zeros(length, dim)
    table[:, 0::2] = torch.sin(position * frequency)
    table[:, 1::2] = torch.cos(position * frequency)
    return table


class TemporalPositionalEncoding(nn.Module):
    """Add a per-frame encoding to ``(B, T, t_s, D)`` tokens."""

    table: torch.Tensor

    def __init__(self, dim: int, max_frames: int = 512) -> None:
        super().__init__()
        self.register_buffer("table", sinusoidal_encoding(max_frames, dim), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Broadcast frame ``j``'s stamp across all spatial tokens of that frame."""
        frames = x.shape[1]
        if frames > self.table.shape[0]:
            raise ValueError(f"{frames} frames exceeds table of {self.table.shape[0]}")
        stamp = self.table[:frames].to(dtype=x.dtype, device=x.device)
        return x + stamp[None, :, None, :]
