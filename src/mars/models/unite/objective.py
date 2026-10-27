"""UNITE objective: weighted CE plus the attention-diversity (AD) loss.

Feature centers C have shape (n_heads, n_frames). The previous center state is
detached; the current batch mean is not, so the between-head term receives a
gradient. With feature_norm l2, P and C are unit-normalised inside the within and
between terms only; the center update stays in raw space.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from mars.losses.ce import cross_entropy_loss
from mars.models.unite.model import UniteOutput


def _stats(
    p_raw: torch.Tensor,
    c_raw: torch.Tensor,
    distance: torch.Tensor,
    gaps: torch.Tensor,
    off_diagonal: torch.Tensor,
    within: torch.Tensor,
    between: torch.Tensor,
) -> dict[str, float]:
    off = gaps[off_diagonal]
    return {
        "ad_within": float(within.detach()),
        "ad_between": float(between.detach()),
        "p_norm": float(p_raw.detach().norm(dim=1).mean()),
        "c_norm": float(c_raw.detach().flatten().norm()),
        "dist_mean": float(distance.detach().mean()),
        "gap_min": float(off.min().detach()) if off.numel() else 0.0,
        "gap_mean": float(off.mean().detach()) if off.numel() else 0.0,
    }


class AttentionDiversityLoss(nn.Module):
    """Maintain per-head centers over pooled features P."""

    centers: torch.Tensor
    delta_within: torch.Tensor

    def __init__(self, n_classes: int, n_heads: int, n_frames: int, cfg: Any) -> None:
        super().__init__()
        self.center_lr = cfg.center_lr
        self.delta_between = cfg.delta_between
        self.between_reduction = getattr(cfg, "between_reduction", "mean")
        self.feature_norm = getattr(cfg, "feature_norm", "l2")
        self.register_buffer("centers", torch.zeros(n_heads, n_frames))
        self.register_buffer("delta_within", torch.tensor(cfg.delta_within, dtype=torch.float32))
        self.last_stats: dict[str, float] = {}

    def current_centers(self, pooled: torch.Tensor) -> torch.Tensor:
        """Blend detached previous centers with this batch's mean."""
        batch_center = pooled.mean(dim=0)
        state = self.centers.detach().to(pooled.dtype)
        return (1.0 - self.center_lr) * state + self.center_lr * batch_center

    def forward(self, pooled: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return within + between hinge terms for this batch."""
        centers = self.current_centers(pooled)
        p_raw = pooled.flatten(1)
        c_raw = centers.flatten()

        if self.feature_norm == "l2":
            flat_p = F.normalize(p_raw, dim=1)
            flat_c = F.normalize(c_raw, dim=0)
            distance = torch.linalg.vector_norm(flat_p - flat_c.unsqueeze(0), dim=1)
            rows = F.normalize(centers, dim=1)
        else:
            distance = torch.linalg.vector_norm(p_raw - c_raw.unsqueeze(0), dim=1)
            rows = centers

        margins = self.delta_within.to(pooled.dtype)[labels]
        within = F.relu(distance - margins).mean()

        gaps = torch.cdist(rows.unsqueeze(0), rows.unsqueeze(0)).squeeze(0)
        off_diagonal = ~torch.eye(rows.shape[0], dtype=torch.bool, device=gaps.device)
        pairs = F.relu(self.delta_between - gaps[off_diagonal])
        between = pairs.sum() if self.between_reduction == "sum" else pairs.mean()

        self.last_stats = _stats(p_raw, c_raw.unsqueeze(0), distance, gaps, off_diagonal, within, between)
        with torch.no_grad():
            self.centers.copy_(centers.detach().to(self.centers.dtype))
        return within + between


class UniteObjective(nn.Module):
    """lambda_ce * CE + lambda_ad * AD."""

    class_weights: torch.Tensor

    def __init__(
        self,
        cfg: Any,
        n_classes: int,
        n_heads: int,
        n_frames: int,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.lambda_ce = cfg.lambda_ce
        self.lambda_ad = cfg.lambda_ad
        self.ad_enabled = cfg.ad.enabled
        self.ce_enabled = getattr(cfg, "ce_enabled", True)
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.empty(0),
        )
        self.ad = AttentionDiversityLoss(n_classes, n_heads, n_frames, cfg.ad)

    def forward(self, output: UniteOutput, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the total loss and detached scalar parts for logging."""
        logits = output.logits.float()
        parts: dict[str, float] = {}
        total = logits.new_zeros(())

        if self.ce_enabled:
            weights = self.class_weights if self.class_weights.numel() else None
            ce = cross_entropy_loss(logits, labels, weights)
            total = total + self.lambda_ce * ce
            parts["ce"] = float(ce.detach())

        if self.ad_enabled:
            assert output.pooled is not None, "AD needs pooled features; run the network in train mode"
            ad = self.ad(output.pooled.float(), labels)
            total = total + self.lambda_ad * ad
            parts["ad"] = float(ad.detach())
            parts.update(self.ad.last_stats)

        parts["total"] = float(total.detach())
        return total, parts
