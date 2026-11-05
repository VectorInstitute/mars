"""Typed configuration for Effort."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mars.config import build_dataclass, load_yaml
from mars.train.config import ModelConfig as BaseModelConfig
from mars.train.config import RunConfig, validate_run


@dataclass
class ModelConfig(BaseModelConfig):
    """CLIP ViT-L/14 whose attention weights train only a low-rank SVD residual."""

    name: str = "effort"
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
    # Singular directions left trainable in every attention projection.
    residual_rank: int = 1
    encoder_chunk: int = 64


@dataclass
class LossConfig:
    """Weight of the mean orthogonality plus singular-value constraint over layers."""

    regularisation: float = 1.0


@dataclass
class Config(RunConfig):
    """Full Effort run configuration."""

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
    if not 1 <= m.residual_rank < m.clip_width:
        raise ValueError(f"residual_rank must lie in 1..{m.clip_width - 1}, got {m.residual_rank}")
    if cfg.loss.regularisation < 0:
        raise ValueError("loss.regularisation must be non-negative")
    validate_run(cfg)
