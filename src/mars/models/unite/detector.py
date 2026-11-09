"""UNITE as a detector: its network and its CE plus attention-diversity loss."""

from __future__ import annotations

from typing import Any, Self

import torch
from torch import nn

from mars.models.base import BaseDetector
from mars.models.unite.config import Config, config_from_dict
from mars.models.unite.model import build_model
from mars.models.unite.objective import UniteObjective


class UniteDetector(BaseDetector):
    """UNITE network with its CE and attention-diversity objective."""

    name = "unite"
    base_model = "google/siglip-so400m-patch14-384"
    cfg: Config

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> Self:
        """Build from a merged YAML dict."""
        return cls(config_from_dict(raw))

    def build_network(self) -> nn.Module:
        """Frozen encoder plus temporal transformer."""
        return build_model(self.cfg.model, self.cfg.data.image_size)

    def build_objective(self, class_weights: torch.Tensor | None) -> nn.Module:
        """Weighted CE plus attention diversity over the pooled features."""
        m = self.cfg.model
        return UniteObjective(self.cfg.loss, m.n_classes, m.n_heads, self.cfg.data.n_frames, class_weights)
