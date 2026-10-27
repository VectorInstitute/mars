"""ForAda objective: cross-entropy per genuine frame.

ForAda's other terms (blending-boundary regression and patch-level contrast) need
face-manipulation masks; every method here trains on video labels only.
"""

from __future__ import annotations

import torch
from torch import nn

from mars.losses.ce import cross_entropy_loss
from mars.models.forada.model import ForAdaOutput


class ForAdaObjective(nn.Module):
    """Mean CE over genuine frames, each carrying its video's label."""

    class_weights: torch.Tensor

    def __init__(self, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights if class_weights is not None else torch.empty(0))

    def forward(self, output: ForAdaOutput, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the loss and its detached value for logging."""
        valid = output.valid
        frame_labels = labels[:, None].expand_as(valid)[valid]
        weights = self.class_weights if self.class_weights.numel() else None
        ce = cross_entropy_loss(output.frame_logits[valid].float(), frame_labels, weights)
        return ce, {"ce": float(ce.detach())}
