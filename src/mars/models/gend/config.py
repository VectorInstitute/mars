"""Typed configuration for GenD."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mars.config import build_dataclass, load_yaml
from mars.train.config import ModelConfig as BaseModelConfig
from mars.train.config import RunConfig, validate_run


@dataclass
class ModelConfig(BaseModelConfig):
    """Perception Encoder with tuned norms and an L2-normalised linear head."""

    name: str = "gend"
    encoder_id: str = "vit_pe_core_large_patch14_336.fb"
    encoder_path: str | None = None
    d_feature: int = 1024
    n_tokens: int = 576
    encoder_chunk: int = 64


@dataclass
class LossConfig:
    """Weights of the cross-entropy, uniformity and label-aware alignment terms."""

    ce: float = 1.0
    uniformity: float = 0.5
    alignment: float = 0.1


@dataclass
class Config(RunConfig):
    """Full GenD run configuration."""

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
    if "pe_core" not in cfg.model.encoder_id.lower():
        raise ValueError(
            f"GenD needs a Perception Encoder (encoder_id containing pe_core), got {cfg.model.encoder_id!r}"
        )
    if min(cfg.loss.ce, cfg.loss.uniformity, cfg.loss.alignment) < 0:
        raise ValueError("loss weights must be non-negative")
    validate_run(cfg)
