"""MARS loss: cross-entropy on the output, plus one per path when there are two."""

from __future__ import annotations

import torch
from torch import nn

from mars.losses.ce import cross_entropy_loss
from mars.models.mars.residual_path import MarsOutput


class MarsObjective(nn.Module):
    """CE on the fused logits, which trains only the gate, plus CE on each path."""

    class_weights: torch.Tensor

    def __init__(self, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights if class_weights is not None else torch.empty(0))

    def forward(self, output: MarsOutput, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the total loss and its detached value for logging."""
        weights = self.class_weights if self.class_weights.numel() else None
        loss = cross_entropy_loss(output.logits.float(), labels, weights)
        for path_logits in output.path_logits:
            loss = loss + cross_entropy_loss(path_logits.float(), labels, weights)
        return loss, {"ce": float(loss.detach())}
