"""Low-rank adaptation of a frozen linear layer.

Initialised as the PEFT library does by default: A Kaiming uniform, B zero, so
the wrapped layer starts out computing exactly what the frozen one does.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus a trainable low-rank update, scaled by alpha / rank.

    Parameters
    ----------
    base : nn.Linear
        Pretrained layer; kept frozen.
    rank : int
        Rank of the update.
    alpha : float
        Scale numerator; the update is multiplied by alpha / rank.
    generator : torch.Generator
        Draws A, so building the adapter leaves the global random stream untouched.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, generator: torch.Generator) -> None:
        super().__init__()
        base.requires_grad_(False)
        self.base = base
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        # Kaiming uniform with a = sqrt(5), as nn.Linear and PEFT initialise A.
        bound = 1 / math.sqrt(base.in_features)
        with torch.no_grad():
            self.lora_a.copy_(torch.rand(self.lora_a.shape, generator=generator) * 2 * bound - bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen output plus the scaled low-rank update."""
        update = (x @ self.lora_a.to(x.dtype).T) @ self.lora_b.to(x.dtype).T
        return self.base(x) + update * self.scale
