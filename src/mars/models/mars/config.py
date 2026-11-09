"""Typed configuration for MARS."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mars.config import build_dataclass, load_yaml
from mars.models.mars.bandpass import scale_pairs
from mars.train.config import ModelConfig as BaseModelConfig
from mars.train.config import RunConfig, validate_run


@dataclass
class BandConfig:
    """One band-pass family: whether it runs, and at which diffusion scales."""

    enabled: bool = True
    scales: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])

    @property
    def bands(self) -> list[tuple[float, float]]:
        """Consecutive scale pairs, one stream each. Empty when disabled."""
        return scale_pairs(self.scales) if self.enabled else []


@dataclass
class StreamsConfig:
    """Which streams the fusion sums, and the operator family behind them."""

    appearance: bool = True
    spatial: BandConfig = field(default_factory=BandConfig)
    temporal: BandConfig = field(default_factory=BandConfig)
    first_difference: bool = True
    operator: str = "dog"

    @property
    def residual_names(self) -> list[str]:
        """Residual streams in the order the fusion builds and sums them."""
        names = [f"spatial_{a:g}-{b:g}" for a, b in self.spatial.bands]
        names += [f"temporal_{a:g}-{b:g}" for a, b in self.temporal.bands]
        if self.first_difference:
            names.append("first_difference")
        return names

    @property
    def n_residual(self) -> int:
        """Number of residual streams."""
        return len(self.residual_names)


@dataclass
class ModelConfig(BaseModelConfig):
    """Frozen encoder, multiscale streams, and a masked temporal transformer."""

    name: str = "mars"
    encoder_id: str = "google/siglip-so400m-patch14-384"
    encoder_path: str | None = None
    d_feature: int = 1152
    n_tokens: int = 729
    grid_side: int = 27
    d_model: int = 1152
    d_resid: int = 256
    depth: int = 4
    n_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    pool_hidden: int = 128
    spatial_pool: str = "attn"
    norm_style: str = "post"
    encoder_chunk: int = 32
    grad_checkpoint: bool = True
    block_norm: bool = False
    temporal_validity: bool = True
    streams: StreamsConfig = field(default_factory=StreamsConfig)
    # Paths: residual (streams + temporal trunk) and semantic (pooled per-frame head).
    encoder_layer: int | None = None
    paths: list[str] = field(default_factory=lambda: ["residual"])
    output: str = "fused"
    semantic_frames: int = 16
    tune_semantic_norms: bool = True
    # Semantic path re-encodes its frames with its own LayerNorms in blocks 1..split;
    # PE, SigLIP or DINOv3, with or without semantic LoRA.
    semantic_early_norms: bool = False
    # LoRA on every linear layer of the semantic blocks; rank 0 keeps them frozen.
    semantic_lora_rank: int = 0
    semantic_lora_alpha: float = 8.0


@dataclass
class Config(RunConfig):
    """Full MARS run configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)


def config_from_dict(raw: dict[str, Any]) -> Config:
    """Build and validate a Config from a merged YAML dict."""
    cfg = build_dataclass(Config, raw)
    validate(cfg)
    return cfg


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config, following an optional _base_ chain."""
    return config_from_dict(load_yaml(path, overrides))


def _validate_model(m: ModelConfig) -> None:
    """Raise ValueError if the network or its streams are inconsistent."""
    s = m.streams
    if m.norm_style not in ("post", "pre"):
        raise ValueError(f"norm_style must be post or pre, got {m.norm_style!r}")
    if m.spatial_pool not in ("mean", "attn"):
        raise ValueError(f"unknown spatial_pool {m.spatial_pool!r}")
    if m.d_model % m.n_heads:
        raise ValueError(f"d_model {m.d_model} not divisible by n_heads {m.n_heads}")
    if m.n_tokens != m.grid_side**2:
        raise ValueError(f"n_tokens {m.n_tokens} is not grid_side {m.grid_side} squared")
    if s.operator not in ("dog", "heat"):
        raise ValueError(f"streams.operator must be dog or heat, got {s.operator!r}")
    # Surfaces a malformed ladder here rather than at the first forward.
    for band in (s.spatial, s.temporal):
        if band.enabled:
            scale_pairs(band.scales)
    if not s.appearance and not s.n_residual:
        raise ValueError("at least one stream must be enabled")
    _validate_paths(m)


def _validate_paths(m: ModelConfig) -> None:
    """Raise ValueError if the path options are inconsistent."""
    if not m.paths or set(m.paths) - {"residual", "semantic"}:
        raise ValueError(f"paths must be a non-empty subset of residual, semantic; got {m.paths}")
    if m.output not in ("fused", *m.paths):
        raise ValueError(f"output {m.output!r} is not fused or one of paths {m.paths}")
    if "semantic" in m.paths and not any(key in m.encoder_id.lower() for key in ("pe_core", "siglip", "dinov3")):
        raise ValueError("the semantic path needs an encoder that can be split (pe_core, siglip or dinov3)")
    if m.semantic_frames < 1:
        raise ValueError(f"semantic_frames must be positive, got {m.semantic_frames}")
    if m.semantic_lora_rank < 0:
        raise ValueError(f"semantic_lora_rank must be non-negative, got {m.semantic_lora_rank}")
    if m.semantic_lora_rank and "semantic" not in m.paths:
        raise ValueError("semantic_lora_rank needs the semantic path")
    if m.semantic_lora_rank and "pe_core" not in m.encoder_id.lower():
        raise ValueError("semantic_lora_rank is implemented for the Perception Encoder only")
    if m.semantic_early_norms:
        if "semantic" not in m.paths:
            raise ValueError("semantic_early_norms needs the semantic path")
        if not m.tune_semantic_norms:
            raise ValueError("semantic_early_norms needs tune_semantic_norms")
        if m.encoder_layer is not None and m.encoder_layer < 1:
            raise ValueError("semantic_early_norms needs encoder_layer >= 1")


def validate(cfg: Config) -> None:
    """Raise ValueError if config combinations are inconsistent."""
    _validate_model(cfg.model)
    validate_run(cfg)
