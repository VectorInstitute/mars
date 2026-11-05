"""Typed configuration for DeMamba."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mars.config import build_dataclass, load_yaml
from mars.train.config import ModelConfig as BaseModelConfig
from mars.train.config import RunConfig, validate_run


@dataclass
class ModelConfig(BaseModelConfig):
    """X-CLIP B/16, fine-tuned, with a Mamba over every spatial zone across time."""

    name: str = "demamba"
    # X-CLIP: a Hugging Face id or local path; pretrained false means random init.
    encoder_id: str = "microsoft/xclip-base-patch16"
    pretrained: bool = True
    patch_size: int = 16
    width: int = 768
    layers: int = 12
    heads: int = 12
    hidden: int = 3072
    message_group: int = 8
    # Zone side in patches; 1 gives one Mamba sequence per patch position.
    zone: int = 1
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4
    drop_path: float = 0.1


@dataclass
class Config(RunConfig):
    """Full DeMamba run configuration."""

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
    grid = cfg.data.image_size // m.patch_size
    if cfg.data.image_size % m.patch_size or grid % m.zone:
        raise ValueError(f"image_size {cfg.data.image_size} must give a patch grid divisible by zone {m.zone}")
    if cfg.data.n_frames % m.message_group:
        raise ValueError(f"n_frames {cfg.data.n_frames} must be a multiple of message_group {m.message_group}")
    validate_run(cfg)
