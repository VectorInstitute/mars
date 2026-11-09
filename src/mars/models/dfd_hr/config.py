"""Typed configuration for DFD-HR."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mars.config import build_dataclass, load_yaml
from mars.train.config import ModelConfig as BaseModelConfig
from mars.train.config import RunConfig, validate_run


@dataclass
class ModelConfig(BaseModelConfig):
    """Frozen CLIP ViT-L/14 with MoE adapters and token and layer routing."""

    name: str = "dfd_hr"
    # CLIP: encoder_path is ViT-L-14.pt and None downloads it;
    # pretrained false leaves CLIP at random init.
    encoder_path: str | None = None
    pretrained: bool = True
    clip_size: int = 224
    clip_patch: int = 14
    clip_width: int = 1024
    clip_layers: int = 24
    clip_heads: int = 16
    clip_dim: int = 768
    # MoE adapter after the attention and the MLP of every block.
    n_experts: int = 4
    top_k: int = 4
    expert_ratio: float = 0.25
    # Blocks (1-based) with layer exit and token routing;
    # the first also feeds the ranking term.
    routed_blocks: list[int] = field(default_factory=lambda: [21, 22, 23])
    capacity: float = 0.75
    router_hidden: int = 64
    query_heads: int = 4
    encoder_chunk: int = 32


@dataclass
class LossConfig:
    """Weight of the token-ranking term next to cross-entropy."""

    ranking: float = 0.1


@dataclass
class Config(RunConfig):
    """Full DFD-HR run configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)


def config_from_dict(raw: dict[str, Any]) -> Config:
    """Build and validate a Config from a merged YAML dict."""
    cfg = build_dataclass(Config, raw)
    validate(cfg)
    return cfg


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config, following an optional _base_ chain."""
    return config_from_dict(load_yaml(path, overrides))


def validate(cfg: Config) -> None:
    """Raise ValueError if config combinations are inconsistent."""
    m = cfg.model
    if cfg.data.image_size != 2 * m.clip_size:
        raise ValueError(f"image_size must be 2 x clip_size ({2 * m.clip_size}): one global view and 2 x 2 tiles")
    if not m.routed_blocks or not set(m.routed_blocks) <= set(range(1, m.clip_layers + 1)):
        raise ValueError(f"routed_blocks {m.routed_blocks} must be non-empty and within 1..{m.clip_layers}")
    if not 0 < m.capacity <= 1:
        raise ValueError(f"capacity must lie in (0, 1], got {m.capacity}")
    if not 1 <= m.top_k <= m.n_experts:
        raise ValueError(f"top_k must lie in 1..n_experts, got {m.top_k}")
    validate_run(cfg)
