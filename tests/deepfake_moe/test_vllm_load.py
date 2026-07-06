"""Unit tests for vLLM loader configuration."""

import os
from unittest.mock import MagicMock

from pytest import MonkeyPatch
from transformers import Qwen3OmniMoeProcessor

from deepfake_moe.models.vllm.qwen_omni_vllm import QwenOmniVLLM
from deepfake_moe.models.vllm.qwen_omni_vllm import _patch_qwen3_processor_for_vllm
from deepfake_moe.models.vllm_config import VllmConfig


def test_vllm_load_uses_v0_and_safe_mm_defaults(monkeypatch: MonkeyPatch) -> None:
    cfg = VllmConfig(
        backend="vllm",
        model_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        tensor_parallel_size=4,
        gpu_memory_utilization=0.85,
        max_model_len=65536,
        max_num_seqs=1,
        audio_sample_rate=16000,
        video_min_frames=16,
        video_max_frames=64,
        use_audio_in_video=False,
        max_new_tokens=256,
    )
    model = QwenOmniVLLM(cfg)

    llm_ctor = MagicMock(return_value=object())
    processor = MagicMock()
    from_pretrained = MagicMock(return_value=processor)
    monkeypatch.setattr("deepfake_moe.models.vllm.qwen_omni_vllm.LLM", llm_ctor)
    monkeypatch.setattr("deepfake_moe.models.vllm.qwen_omni_vllm._assert_supported_vllm_stack", lambda: None)
    monkeypatch.setattr(
        "deepfake_moe.models.vllm.qwen_omni_vllm.Qwen3OmniMoeProcessor.from_pretrained",
        from_pretrained,
    )
    if hasattr(Qwen3OmniMoeProcessor, "_get_num_multimodal_tokens"):
        monkeypatch.delattr(Qwen3OmniMoeProcessor, "_get_num_multimodal_tokens", raising=False)
    monkeypatch.setenv("VLLM_USE_V1", "1")
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)

    loaded = model.load()

    assert loaded is model
    assert os.environ["VLLM_USE_V1"] == "1"
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "fork"
    llm_ctor.assert_called_once()
    kwargs = llm_ctor.call_args.kwargs
    assert kwargs["tensor_parallel_size"] == 4
    assert kwargs["disable_custom_all_reduce"] is True
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["limit_mm_per_prompt"] == {"image": 1, "video": 1, "audio": 1}
    assert kwargs["mm_processor_kwargs"] == {"cache_gb": 0}
    from_pretrained.assert_called_once_with(cfg.model_id)


def test_processor_patch_adds_missing_multimodal_helper() -> None:
    _patch_qwen3_processor_for_vllm()
    assert hasattr(Qwen3OmniMoeProcessor, "_get_num_multimodal_tokens")
