"""DFD-HR objective: per-frame cross-entropy plus 0.1 times the token-ranking term."""

from __future__ import annotations

import torch
from torch import nn

from mars.losses.ce import cross_entropy_loss
from mars.models.dfd_hr.config import LossConfig
from mars.models.dfd_hr.model import DFDHROutput


class DFDHRObjective(nn.Module):
    """CE over genuine frames, each carrying its video's label, plus ranking."""

    class_weights: torch.Tensor

    def __init__(self, cfg: LossConfig, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("class_weights", class_weights if class_weights is not None else torch.empty(0))

    def forward(self, output: DFDHROutput, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the total loss and its terms for logging."""
        valid = output.valid
        frame_labels = labels[:, None].expand_as(valid)[valid]
        weights = self.class_weights if self.class_weights.numel() else None
        ce = cross_entropy_loss(output.frame_logits[valid].float(), frame_labels, weights)
        total = ce + self.cfg.ranking * output.ranking
        return total, {"ce": float(ce.detach()), "ranking": float(output.ranking.detach())}
