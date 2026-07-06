"""Unit tests for multi-GPU input placement in QwenOmni."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from deepfake_moe.models.hf.qwen_omni import QwenOmni


def _make_qwen() -> QwenOmni:
    qwen = QwenOmni.__new__(QwenOmni)
    qwen.config = MagicMock()
    qwen.processor = MagicMock()
    return qwen


def test_tensor_device_routing() -> None:
    devices = {
        "default": torch.device("cuda:0"),
        "visual": torch.device("cuda:1"),
        "audio": torch.device("cuda:2"),
    }
    assert QwenOmni._tensor_device("input_ids", devices) == devices["default"]
    assert QwenOmni._tensor_device("pixel_values_videos", devices) == devices["visual"]
    assert QwenOmni._tensor_device("video_grid_thw", devices) == devices["visual"]
    assert QwenOmni._tensor_device("input_features", devices) == devices["audio"]
    assert QwenOmni._tensor_device("feature_attention_mask", devices) == devices["audio"]


def test_thinker_input_devices_from_hf_map() -> None:
    qwen = _make_qwen()
    embed = SimpleNamespace(parameters=lambda: iter([]))
    visual = SimpleNamespace(parameters=lambda: iter([]))
    audio = SimpleNamespace(parameters=lambda: iter([]))
    thinker = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed), visual=visual, audio_tower=audio)
    model = SimpleNamespace(
        thinker=thinker,
        hf_device_map={
            "thinker.model.embed_tokens": 0,
            "thinker.visual": 1,
            "thinker.audio_tower": 2,
        },
        dtype=torch.float16,
    )
    qwen.model = model

    devices = qwen._thinker_input_devices(model)
    assert devices["default"] == torch.device("cuda:0")
    assert devices["visual"] == torch.device("cuda:1")
    assert devices["audio"] == torch.device("cuda:2")
