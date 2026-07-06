"""Named-model registry backed by ``configs/models.yaml``.

Each top-level key in the registry is a **model name** (used in result paths and
selectable via ``--model NAME``); its value is a model entry (see
:mod:`deepfake_moe.models.factory`). Add a new model by adding an entry — no code
changes required.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from deepfake_moe.models.base import Predictor
from deepfake_moe.models.config import Config
from deepfake_moe.models.factory import build_config, build_predictor
from deepfake_moe.models.vllm_config import VllmConfig


DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[3] / "configs" / "models.yaml"


def load_registry(path: str | Path | None = None) -> dict[str, dict]:
    """Return the ``{model_name: entry}`` mapping from the registry YAML."""
    registry_path = Path(path or DEFAULT_REGISTRY_PATH)
    data = yaml.safe_load(registry_path.read_text()) or {}
    if isinstance(data.get("models"), dict):  # optional top-level `models:` wrapper
        data = data["models"]
    if not isinstance(data, dict) or not data:
        raise ValueError(f"No model entries found in registry: {registry_path}")
    return data


def _entry(model_name: str, path: str | Path | None = None) -> dict:
    registry = load_registry(path)
    if model_name not in registry:
        raise ValueError(f"Model {model_name!r} not in registry. Available: {sorted(registry)}")
    return registry[model_name]


def load_config(model_name: str, path: str | Path | None = None) -> Config | VllmConfig:
    """Return the typed config for ``model_name`` (no model weights loaded)."""
    return build_config(_entry(model_name, path))


def load_model(model_name: str, path: str | Path | None = None) -> tuple[Predictor, Config | VllmConfig]:
    """Return ``(model, config)`` for ``model_name`` from the registry."""
    cfg = load_config(model_name, path)
    return build_predictor(cfg), cfg
