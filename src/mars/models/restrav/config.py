"""Typed configuration for ReStraV."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mars.config import build_dataclass, load_yaml
from mars.train.config import ModelConfig as BaseModelConfig
from mars.train.config import RunConfig, validate_run


@dataclass
class ModelConfig(BaseModelConfig):
    """Frozen DINOv2 trajectory geometry and a small MLP."""

    name: str = "restrav"
    # torch.hub repository and model, as released.
    encoder_repo: str = "facebookresearch/dinov2"
    encoder_id: str = "dinov2_vits14"
    # Leading steps kept individually: distances and turning angles.
    early_steps: int = 7
    early_angles: int = 6
    hidden: list[int] = field(default_factory=lambda: [64, 32])
    encoder_chunk: int = 64


@dataclass
class Config(RunConfig):
    """Full ReStraV run configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)


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
    if m.early_angles >= m.early_steps or m.early_steps >= cfg.data.n_frames:
        raise ValueError("need early_angles < early_steps < n_frames")
    validate_run(cfg)
