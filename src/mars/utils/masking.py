"""Averages over genuine frames: padded frames never count."""

from __future__ import annotations

import torch


def masked_mean(x: torch.Tensor, valid: torch.Tensor, dim: int) -> torch.Tensor:
    """Mean of x over dim, counting only positions where valid is True.

    Parameters
    ----------
    x : torch.Tensor
        Values, e.g. (B, T, C) frame logits.
    valid : torch.Tensor
        Boolean mask with the leading shape of x up to and including dim, e.g.
        (B, T); trailing dimensions of x broadcast.
    dim : int
        Dimension to average over.

    Returns
    -------
    torch.Tensor
        x reduced over dim; a row with no valid position gives zeros.
    """
    weight = valid.to(x.dtype).reshape(*valid.shape, *([1] * (x.dim() - valid.dim())))
    return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp(min=1.0)
