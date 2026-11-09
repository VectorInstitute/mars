"""Typed configuration for ForAda."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mars.config import build_dataclass, load_yaml
from mars.train.config import ModelConfig as BaseModelConfig
from mars.train.config import RunConfig, validate_run


@dataclass
class ModelConfig(BaseModelConfig):
    """Frozen CLIP ViT-L/14 steered by a trainable ViT-Tiny adapter."""

    name: str = "forada"
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
    # Adapter: ViT-Tiny blocks over the full-resolution frame.
    adapter_id: str = "vit_tiny_patch16_224"
    adapter_patch: int = 16
    adapter_blocks: int = 8
    n_queries: int = 128
    # Adapter block -> CLIP block whose patch tokens are fused after it.
    fusion_map: dict[int, int] = field(default_factory=lambda: {1: 1, 2: 8, 3: 15})
    # CLIP block after which the trainable intra block adds its residual.
    intra_block: int = 7
    mlp_dim: int = 256
    mlp_out_dim: int = 128
    encoder_chunk: int = 64


@dataclass
class Config(RunConfig):
    """Full ForAda run configuration."""

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
    clip_grid, adapter_grid = m.clip_size // m.clip_patch, cfg.data.image_size // m.adapter_patch
    if clip_grid != adapter_grid:
        raise ValueError(f"CLIP grid {clip_grid} and adapter grid {adapter_grid} must match (the bias spans both)")
    if not set(m.fusion_map) <= set(range(1, m.adapter_blocks + 1)):
        raise ValueError(f"fusion_map blocks {sorted(m.fusion_map)} exceed adapter_blocks {m.adapter_blocks}")
    if not 1 <= m.intra_block <= m.clip_layers or max(m.fusion_map.values()) > m.clip_layers:
        raise ValueError("intra_block and fusion_map CLIP layers must lie within clip_layers")
    if m.n_queries % 4:
        raise ValueError(f"n_queries must be divisible by 4, got {m.n_queries}")
    validate_run(cfg)
