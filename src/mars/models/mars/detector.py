"""MARS as a detector: its network, its loss, and its stream diagnostics."""

from __future__ import annotations

from typing import Any, Self

import torch
from torch import nn

from mars.models.base import BaseDetector
from mars.models.mars.config import Config, config_from_dict
from mars.models.mars.model import build_model
from mars.models.mars.objective import MarsObjective


class MarsDetector(BaseDetector):
    """MARS network with its cross-entropy objective."""

    name = "mars"
    base_model = "timm/vit_pe_core_large_patch14_336.fb"
    cfg: Config

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> Self:
        """Build from a merged YAML dict."""
        return cls(config_from_dict(raw))

    def build_network(self) -> nn.Module:
        """MARS, or single-path MARS when paths is residual only."""
        return build_model(self.cfg.model, self.cfg.data.n_frames, self.cfg.data.image_size)

    def build_objective(self, class_weights: torch.Tensor | None) -> nn.Module:
        """Cross-entropy on the output and on each path."""
        return MarsObjective(class_weights)

    def train_stats(self, network: nn.Module) -> dict[str, float]:
        """RMS of each residual stream entering the fusion."""
        fusion = getattr(network, "fusion", None)
        return {f"rms/{name}": value for name, value in fusion.stream_rms.items()} if fusion is not None else {}
