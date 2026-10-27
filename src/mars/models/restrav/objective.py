"""ReStraV objective: binary cross-entropy on the clip score.

The network emits logits (0, score), so cross-entropy over them equals the released
BCE-with-logits on score.
"""

from __future__ import annotations

import torch
from torch import nn

from mars.losses.ce import cross_entropy_loss
from mars.models.restrav.model import ReStraVOutput


class ReStraVObjective(nn.Module):
    """Cross-entropy over clips."""

    class_weights: torch.Tensor

    def __init__(self, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights if class_weights is not None else torch.empty(0))

    def forward(self, output: ReStraVOutput, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the loss and its detached value for logging."""
        weights = self.class_weights if self.class_weights.numel() else None
        ce = cross_entropy_loss(output.logits.float(), labels, weights)
        return ce, {"ce": float(ce.detach())}
