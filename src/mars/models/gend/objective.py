"""GenD objective: per-frame cross-entropy, uniformity and label-aware alignment.

Every genuine frame is one sample carrying its video's label, as GenD trains on
single images. Uniformity and alignment follow Wang and Isola (ICML 2020) on the
unit-norm embeddings.
"""

from __future__ import annotations

import torch
from torch import nn

from mars.losses.ce import cross_entropy_loss
from mars.models.gend.config import LossConfig
from mars.models.gend.model import GenDOutput


def squared_distances(x: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distances of (N, D) rows, (N, N).

    Computed without a square root, so the gradient stays finite for identical rows.
    """
    norms = x.pow(2).sum(dim=1)
    return (norms[:, None] + norms[None, :] - 2 * x @ x.T).clamp(min=0)


def uniformity(x: torch.Tensor, t: float = 2.0, floor: float = 1e-6) -> torch.Tensor:
    """Log of the mean Gaussian potential over distinct pairs of unit-norm rows."""
    pairs = torch.triu_indices(len(x), len(x), offset=1, device=x.device)
    distances = squared_distances(x)[pairs[0], pairs[1]]
    return distances.mul(-t).exp().mean().clamp(min=floor).log()


def alignment(x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean squared distance over distinct pairs of unit-norm rows sharing a label."""
    same = (labels[:, None] == labels[None, :]).triu(diagonal=1)
    if not same.any():
        return x.new_zeros(())
    return squared_distances(x)[same].mean()


class GenDObjective(nn.Module):
    """ce * CE + uniformity * U + alignment * A over genuine frames."""

    class_weights: torch.Tensor

    def __init__(self, cfg: LossConfig, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("class_weights", class_weights if class_weights is not None else torch.empty(0))

    def forward(self, output: GenDOutput, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the total loss and its detached terms for logging."""
        valid = output.valid
        frame_labels = labels[:, None].expand_as(valid)[valid]
        logits = output.frame_logits[valid].float()
        embeddings = output.embeddings[valid].float()
        weights = self.class_weights if self.class_weights.numel() else None

        ce = cross_entropy_loss(logits, frame_labels, weights)
        spread = uniformity(embeddings)
        pull = alignment(embeddings, frame_labels)
        total = self.cfg.ce * ce + self.cfg.uniformity * spread + self.cfg.alignment * pull
        parts = {"ce": float(ce.detach()), "uniformity": float(spread.detach()), "alignment": float(pull.detach())}
        return total, parts
