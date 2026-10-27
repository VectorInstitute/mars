"""Initialisation helpers that need their own random stream."""

from __future__ import annotations

import math

import torch
from torch import nn


def reseed_linear(layer: nn.Linear, seed: int) -> int:
    """Re-draw layer from a generator seeded with seed; return its size.

    A layer's starting values come from the global stream in build order, so any
    change to an earlier module shifts them. Drawing the final classifier from its
    own stream instead keeps it identical across variants, leaving only the change
    under test.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        weight = torch.empty(layer.weight.shape, dtype=torch.float32)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5), generator=generator)
        layer.weight.copy_(weight.to(layer.weight.device))
        size = layer.weight.numel()
        if layer.bias is not None:
            bound = 1.0 / math.sqrt(layer.weight.shape[1])
            bias = torch.empty(layer.bias.shape, dtype=torch.float32).uniform_(-bound, bound, generator=generator)
            layer.bias.copy_(bias.to(layer.bias.device))
            size += layer.bias.numel()
    return size
