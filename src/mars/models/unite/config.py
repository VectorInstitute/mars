"""Typed configuration for Unite."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mars.config import build_dataclass, load_yaml
from mars.train.config import ModelConfig as BaseModelConfig
from mars.train.config import RunConfig, validate_run


@dataclass
class ModelConfig(BaseModelConfig):
    """Frozen encoder plus temporal transformer."""

    name: str = "unite"
    encoder_id: str = "google/siglip-so400m-patch14-384"
    encoder_path: str | None = None
    token_layout: str = "per_token_temporal"
    spatial_pool: str = "mean"
    pool_hidden: int = 128
    d_feature: int = 1152
    n_tokens: int = 729
    d_model: int = 1152
    depth: int = 4
    n_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    encoder_chunk: int = 32
    norm_style: str = "post"
    grad_checkpoint: bool = True
    input_norm: bool = False


@dataclass
class ADLossConfig:
    """Attention-diversity loss hyperparameters."""

    enabled: bool = True
    center_lr: float = 0.05
    delta_within: list[float] = field(default_factory=lambda: [0.01, -2.0])
    delta_between: float = 0.5
    between_reduction: str = "mean"
    feature_norm: str = "l2"


@dataclass
class LossConfig:
    """Weighted combination of CE and AD losses."""

    lambda_ce: float = 0.5
    lambda_ad: float = 0.5
    ce_enabled: bool = True
    ad: ADLossConfig = field(default_factory=ADLossConfig)


@dataclass
class Config(RunConfig):
    """Full Unite run configuration."""

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
    if m.norm_style not in ("post", "pre"):
        raise ValueError(f"norm_style must be post or pre, got {m.norm_style!r}")
    if m.token_layout != "per_token_temporal":
        raise ValueError(f"unknown token_layout {m.token_layout!r}")
    if m.spatial_pool not in ("mean", "attn"):
        raise ValueError(f"unknown spatial_pool {m.spatial_pool!r}")
    if m.d_model % m.n_heads:
        raise ValueError(f"d_model {m.d_model} not divisible by n_heads {m.n_heads}")
    if len(cfg.loss.ad.delta_within) != m.n_classes:
        raise ValueError(f"delta_within has {len(cfg.loss.ad.delta_within)} entries but n_classes is {m.n_classes}")
    if cfg.loss.ad.feature_norm not in ("none", "l2"):
        raise ValueError(f"feature_norm must be none or l2, got {cfg.loss.ad.feature_norm!r}")
    validate_run(cfg)
