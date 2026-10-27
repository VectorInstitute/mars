"""Abstract detector: the part of a run that belongs to one method."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

import torch
from torch import nn

from mars.train.config import RunConfig


class BaseDetector(ABC):
    """A method's network and objective; the shared trainer and evaluator do the rest.

    Attributes
    ----------
    name : str
        Registry key, matching model.name in the YAML.
    cfg : RunConfig
        The full run configuration.
    dynamic_graph : bool
        True when a step may skip trainable parts of the network, which
        distributed training must then be told to expect.
    base_model : str
        Pretrained model the method builds on, recorded in footprint reports.
    relation : str
        How the method relates to base_model: adapter, lora or finetune.
    """

    name: str
    dynamic_graph: bool = False
    base_model: str = ""
    relation: str = "adapter"

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg

    @classmethod
    @abstractmethod
    def from_config(cls, raw: dict[str, Any]) -> Self:
        """Build an instance from a merged YAML dict."""

    @abstractmethod
    def build_network(self) -> nn.Module:
        """Build the network.

        Its forward takes frames (B, T, 3, S, S) and valid (B, T), the mask of
        genuine frames, and returns an output with logits (B, C). Scoring frames
        or the whole clip is the network's own choice. It exposes classifier,
        its final linear layer, which training re-draws from the seed.
        """

    @abstractmethod
    def build_objective(self, class_weights: torch.Tensor | None) -> nn.Module:
        """Build the loss, mapping (output, labels) to (loss, parts).

        parts is a dict of detached floats logged each step. Buffers the loss
        keeps across steps are averaged over ranks after every update.
        """

    def train_stats(self, network: nn.Module) -> dict[str, float]:
        """Extra scalars to log each step; none by default."""
        return {}
