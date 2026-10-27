"""DeMamba as a detector: its network and its cross-entropy."""

from __future__ import annotations

from typing import Any, Self

import torch
from torch import nn

from mars.models.base import BaseDetector
from mars.models.demamba.config import Config, config_from_dict
from mars.models.demamba.model import build_model
from mars.models.demamba.objective import DeMambaObjective


class DeMambaDetector(BaseDetector):
    """DeMamba network with clip-level cross-entropy."""

    name = "demamba"
    base_model = "microsoft/xclip-base-patch16"
    relation = "finetune"
    cfg: Config

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> Self:
        """Build from a merged YAML dict."""
        return cls(config_from_dict(raw))

    def build_network(self) -> nn.Module:
        """X-CLIP with the global and zone-wise Mamba branches."""
        return build_model(self.cfg.model, self.cfg.data.image_size)

    def build_objective(self, class_weights: torch.Tensor | None) -> nn.Module:
        """Clip-level cross-entropy."""
        return DeMambaObjective(class_weights)
