"""YAML-backed vLLM eval configuration."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from deepfake_moe.utils.env import expand_env


@dataclass
class VllmConfig:
    """Typed view over a ``vllm``-backend model entry (eval inference only)."""

    backend: str
    model_id: str
    tensor_parallel_size: int | None
    gpu_memory_utilization: float
    max_model_len: int
    max_num_seqs: int
    audio_sample_rate: int
    video_min_frames: int
    video_max_frames: int
    use_audio_in_video: bool
    max_new_tokens: int
    limit_mm_per_prompt: dict[str, int] | None = None
    mm_processor_kwargs: dict[str, Any] | None = None
    enable_prefix_caching: bool = False
    seed: int = 1234
    worker_multiproc_method: str = "spawn"

    @classmethod
    def from_dict(cls, data: dict) -> VllmConfig:
        """Build from a mapping, expanding ``${VAR}`` and rejecting unknown keys."""
        data = expand_env(dict(data))
        if data.get("backend") != "vllm":
            raise ValueError(f"Not a vLLM config (backend={data.get('backend')!r})")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown vLLM config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> VllmConfig:
        """Load a single vLLM model config from YAML."""
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})
