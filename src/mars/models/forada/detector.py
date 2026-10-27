"""ForAda as a detector: its network and its per-frame CE."""

from __future__ import annotations

from typing import Any, Self

import torch
from torch import nn

from mars.models.base import BaseDetector
from mars.models.forada.config import Config, config_from_dict
from mars.models.forada.model import build_model
from mars.models.forada.objective import ForAdaObjective


class ForAdaDetector(BaseDetector):
    """ForAda network with per-frame cross-entropy."""

    name = "forada"
    base_model = "openai/clip-vit-large-patch14"
    cfg: Config

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> Self:
        """Build from a merged YAML dict."""
        return cls(config_from_dict(raw))

    def build_network(self) -> nn.Module:
        """Frozen CLIP, the adapter, the guided pass and the query head."""
        return build_model(self.cfg.model)

    def build_objective(self, class_weights: torch.Tensor | None) -> nn.Module:
        """Per-frame cross-entropy."""
        return ForAdaObjective(class_weights)
