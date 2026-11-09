"""Effort as a detector: its network and its CE plus SVD constraints."""

from __future__ import annotations

from typing import Any, Self

import torch
from torch import nn

from mars.models.base import BaseDetector
from mars.models.effort.config import Config, config_from_dict
from mars.models.effort.model import build_model
from mars.models.effort.objective import EffortObjective


class EffortDetector(BaseDetector):
    """Effort network with per-frame cross-entropy and the SVD constraints."""

    name = "effort"
    base_model = "openai/clip-vit-large-patch14"
    relation = "lora"
    cfg: Config

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> Self:
        """Build from a merged YAML dict."""
        return cls(config_from_dict(raw))

    def build_network(self) -> nn.Module:
        """Frozen CLIP with SVD-residual attention projections and a linear head."""
        return build_model(self.cfg.model)

    def build_objective(self, class_weights: torch.Tensor | None) -> nn.Module:
        """Per-frame cross-entropy plus the weighted SVD constraints."""
        return EffortObjective(self.cfg.loss, class_weights)
