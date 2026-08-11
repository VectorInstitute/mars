"""Tests for YAML config loaders."""

from pathlib import Path

import pytest
import yaml

from deepfake_moe.eval.dataset import EvalConfig
from deepfake_moe.models.config import Config
from deepfake_moe.models.registry import load_config


def test_config_from_registry_hf_entry() -> None:
    cfg = load_config("qwen3-omni-hf")
    assert isinstance(cfg, Config)
    assert cfg.audio_sample_rate == 16_000
    assert cfg.max_new_tokens > 0
    assert cfg.model_id


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    base = load_config("qwen3-omni-hf")
    assert isinstance(base, Config)
    data = {
        "model_id": base.model_id,
        "device_map": base.device_map,
        "dtype": base.dtype,
        "attn_implementation": base.attn_implementation,
        "audio_sample_rate": base.audio_sample_rate,
        "video_fps": base.video_fps,
        "video_min_frames": base.video_min_frames,
        "video_max_frames": base.video_max_frames,
        "use_audio_in_video": base.use_audio_in_video,
        "max_new_tokens": base.max_new_tokens,
        "unexpected_key": True,
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="Unknown config keys"):
        Config.from_yaml(path)


def test_eval_config_from_yaml() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "configs" / "eval.yaml"
    cfg = EvalConfig.from_yaml(path)
    assert cfg.num_frames == 32
    assert "lavdf" in cfg.dataset_sources
    assert cfg.model
    assert cfg.experiment_name
    assert cfg.manifest_dir
    assert cfg.data_root
    assert cfg.dataset_video_roots
    assert "lavdf" in cfg.dataset_video_roots
    assert cfg.resolved_audio_dir().endswith("processed/audio/normalized")
