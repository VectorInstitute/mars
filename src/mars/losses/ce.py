"""Cross-entropy classification loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean cross-entropy over a batch.

    Parameters
    ----------
    logits : torch.Tensor
        Unnormalized scores, shape ``(N, C)``.
    labels : torch.Tensor
        Class indices, shape ``(N,)``.
    class_weights : torch.Tensor or None
        Optional per-class weights.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    return F.cross_entropy(logits, labels, weight=class_weights)
