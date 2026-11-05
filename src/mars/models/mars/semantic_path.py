"""MARS semantic path: a per-frame linear head on the encoder's pooled output."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SemanticPath(nn.Module):
    """Per-frame linear head on L2-normalised pooled features, averaged over frames."""

    def __init__(self, d_feature: int, n_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(d_feature, n_classes)

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map (B, S, D) pooled features to a (B, D) embedding and (B, C) logits."""
        unit = F.normalize(pooled.float(), dim=-1)
        return unit.mean(dim=1), self.classifier(unit).mean(dim=1)


def frame_subset(valid: torch.Tensor, count: int) -> torch.Tensor:
    """(B, count) indices spread evenly over each clip's genuine frames."""
    n_valid = valid.sum(dim=1)
    steps = torch.linspace(0.0, 1.0, count, device=valid.device)
    return (steps[None, :] * (n_valid[:, None] - 1).clamp(min=0)).round().long()
