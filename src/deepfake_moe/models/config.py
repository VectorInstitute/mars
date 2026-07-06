"""YAML-backed run configuration."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from deepfake_moe.utils.env import expand_env


@dataclass
class Config:
    """Typed view over an ``hf``-backend model entry (values live in YAML)."""

    model_id: str
    device_map: str
    dtype: str
    attn_implementation: str | None

    audio_sample_rate: int

    video_fps: float
    video_min_frames: int
    video_max_frames: int
    use_audio_in_video: bool

    max_new_tokens: int

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Build from a mapping, expanding ``${VAR}`` and rejecting unknown keys."""
        data = expand_env(dict(data))
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load a single-model config from YAML."""
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})
