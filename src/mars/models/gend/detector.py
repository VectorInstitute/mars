"""GenD as a detector: its network and its CE, uniformity and alignment loss."""

from __future__ import annotations

from typing import Any, Self

import torch
from torch import nn

from mars.models.base import BaseDetector
from mars.models.gend.config import Config, config_from_dict
from mars.models.gend.model import build_model
from mars.models.gend.objective import GenDObjective


class GenDDetector(BaseDetector):
    """GenD network with its per-frame objective."""

    name = "gend"
    base_model = "timm/vit_pe_core_large_patch14_336.fb"
    relation = "finetune"
    cfg: Config

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> Self:
        """Build from a merged YAML dict."""
        return cls(config_from_dict(raw))

    def build_network(self) -> nn.Module:
        """Perception Encoder with tuned norms and an L2-normalised head."""
        return build_model(self.cfg.model, self.cfg.data.image_size)

    def build_objective(self, class_weights: torch.Tensor | None) -> nn.Module:
        """Per-frame CE, uniformity and label-aware alignment."""
        return GenDObjective(self.cfg.loss, class_weights)
