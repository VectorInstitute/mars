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

    def forward(self, x: torch.Tensor, index: torch.Tensor | None = None) -> torch.Tensor:
        """Stamp every spatial token of a frame with that frame's encoding.

        index, (B, T), reads the table at given positions instead of 0..T-1, so a
        frame can be stamped by its order among genuine frames.
        """
        if index is None:
            index = torch.arange(x.shape[1], device=x.device)[None, :]
        if int(index.max()) >= self.table.shape[0]:
            raise ValueError(f"position {int(index.max())} exceeds table of {self.table.shape[0]}")
        stamp = self.table[index].to(dtype=x.dtype, device=x.device)
        return x + stamp[:, :, None, :]
