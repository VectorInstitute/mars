"""Tests for the model registry, config builder, and predictor factory."""

from unittest.mock import patch

import pytest

from deepfake_moe.models.factory import build_config
from deepfake_moe.models.registry import load_config, load_model
from deepfake_moe.models.vllm_config import VllmConfig


def test_registry_loads_vllm_entry() -> None:
    cfg = load_config("qwen3-omni")
    assert isinstance(cfg, VllmConfig)
    assert cfg.backend == "vllm"
    assert cfg.tensor_parallel_size is None


def test_registry_unknown_model() -> None:
    with pytest.raises(ValueError, match="not in registry"):
        load_config("does-not-exist")


def test_build_config_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        build_config({"backend": "unknown", "model_id": "x"})


def test_registry_selects_vllm_backend() -> None:
    class _FakeVLLM:
        def __init__(self, cfg: VllmConfig) -> None:
            self.config = cfg

        def load(self) -> "_FakeVLLM":
            return self

        def predict(self, clip: object) -> None:
            return None

    with patch("deepfake_moe.models.vllm.qwen_omni_vllm.QwenOmniVLLM", _FakeVLLM):
        model, cfg = load_model("qwen3-omni")

    assert isinstance(cfg, VllmConfig)
    assert cfg.backend == "vllm"
    assert model.config is cfg  # type: ignore[attr-defined]
