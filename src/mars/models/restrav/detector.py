"""ReStraV as a detector: its network and its cross-entropy."""

from __future__ import annotations

from typing import Any, Self

import torch
from torch import nn

from mars.models.base import BaseDetector
from mars.models.restrav.config import Config, config_from_dict
from mars.models.restrav.model import build_model
from mars.models.restrav.objective import ReStraVObjective


class ReStraVDetector(BaseDetector):
    """ReStraV network with clip-level cross-entropy."""

    name = "restrav"
    base_model = "facebook/dinov2-small"
    cfg: Config

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> Self:
        """Build from a merged YAML dict."""
        return cls(config_from_dict(raw))

    def build_network(self) -> nn.Module:
        """Frozen DINOv2 with trajectory features and an MLP."""
        return build_model(self.cfg.model)

    def build_objective(self, class_weights: torch.Tensor | None) -> nn.Module:
        """Clip-level cross-entropy."""
        return ReStraVObjective(class_weights)
