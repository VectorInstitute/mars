"""Effort objective: per-frame cross-entropy plus the weighted SVD constraints.

The constraints follow the paper (orthogonality and singular-value terms); the
released trainers carry them commented out, with weight 1.0 on their mean.
"""

from __future__ import annotations

import torch
from torch import nn

from mars.losses.ce import cross_entropy_loss
from mars.models.effort.config import LossConfig
from mars.models.effort.model import EffortOutput


class EffortObjective(nn.Module):
    """CE over genuine frames, each carrying its video's label, plus regularisation."""

    class_weights: torch.Tensor

    def __init__(self, cfg: LossConfig, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("class_weights", class_weights if class_weights is not None else torch.empty(0))

    def forward(self, output: EffortOutput, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the total loss and its terms for logging."""
        valid = output.valid
        frame_labels = labels[:, None].expand_as(valid)[valid]
        weights = self.class_weights if self.class_weights.numel() else None
        ce = cross_entropy_loss(output.frame_logits[valid].float(), frame_labels, weights)
        total = ce + self.cfg.regularisation * output.regularisation
        return total, {"ce": float(ce.detach()), "regularisation": float(output.regularisation.detach())}
