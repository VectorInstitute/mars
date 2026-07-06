"""Build a typed model config and a ``Predictor`` from a model entry.

A model entry is a plain mapping (one value in ``configs/models.yaml``) whose
``backend`` field selects the loader: ``hf`` -> :class:`Config` + ``QwenOmni``,
``vllm`` -> :class:`VllmConfig` + ``QwenOmniVLLM``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from deepfake_moe.models.base import Predictor
from deepfake_moe.models.config import Config
from deepfake_moe.models.vllm_config import VllmConfig


def build_config(entry: dict) -> Config | VllmConfig:
    """Return the typed config for one model entry, dispatching on ``backend``."""
    backend = entry.get("backend", "hf")
    if backend == "vllm":
        return VllmConfig.from_dict(entry)
    if backend == "hf":
        return Config.from_dict({k: v for k, v in entry.items() if k != "backend"})
    raise ValueError(f"Unknown backend {backend!r} in model config")


def build_predictor(cfg: Config | VllmConfig) -> Predictor:
    """Instantiate and load the backend implementation for ``cfg``."""
    if getattr(cfg, "backend", "hf") == "vllm":
        from deepfake_moe.models.vllm.qwen_omni_vllm import QwenOmniVLLM  # noqa: PLC0415

        return QwenOmniVLLM(cfg).load()  # type: ignore[arg-type]

    from deepfake_moe.models.hf.qwen_omni import QwenOmni  # noqa: PLC0415

    return QwenOmni(cfg).load()  # type: ignore[arg-type]


def load_predictor(path: str | Path) -> tuple[Predictor, Config | VllmConfig]:
    """Load ``(model, config)`` from a single-model YAML file."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    cfg = build_config(data)
    return build_predictor(cfg), cfg
