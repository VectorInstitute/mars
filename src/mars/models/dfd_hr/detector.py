"""DFD-HR as a detector: its network and its CE plus ranking loss."""

from __future__ import annotations

from typing import Any, Self

import torch
from torch import nn

from mars.models.base import BaseDetector
from mars.models.dfd_hr.config import Config, config_from_dict
from mars.models.dfd_hr.model import build_model
from mars.models.dfd_hr.objective import DFDHRObjective


class DFDHRDetector(BaseDetector):
    """DFD-HR network with per-frame cross-entropy and the ranking term."""

    name = "dfd_hr"
    base_model = "openai/clip-vit-large-patch14"
    cfg: Config
    # A view that exits early skips the routed blocks, and their token routers with it.
    dynamic_graph = True

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> Self:
        """Build from a merged YAML dict."""
        return cls(config_from_dict(raw))

    def build_network(self) -> nn.Module:
        """Frozen CLIP with MoE adapters, routing and the two-view head."""
        return build_model(self.cfg.model)

    def build_objective(self, class_weights: torch.Tensor | None) -> nn.Module:
        """Per-frame cross-entropy plus the token-ranking term."""
        return DFDHRObjective(self.cfg.loss, class_weights)
