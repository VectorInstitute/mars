"""Tests for manifest loading and per-clip media resolution."""

import json
from pathlib import Path

import numpy as np
import pytest

from deepfake_moe.eval.dataset import EvalConfig, load_clip, load_manifest_entries


def _entry(clip_id: str, dataset: str = "tiny") -> dict:
    return {
        "clip_id": clip_id,
        "dataset": dataset,
        "split": "train",
        "unified_split": "train",
        "label": [0, 0],
        "label_name": "real",
        "label_source": "metadata",
        "src_path": f"/fake/{clip_id}.mp4",
        "wav_path": None,
        "has_audio": False,
        "duration_seconds": 1.0,
        "fake_periods": [],
        "fake_ratio": 0.0,
        "language": None,
        "audio_model": None,
        "video_model": None,
    }


def _cfg(manifest_dir: Path, out_dir: Path, **overrides) -> EvalConfig:
    base = {
        "manifest_dir": str(manifest_dir),
        "dataset_sources": {"tiny": ["tiny_train.jsonl"]},
        "num_clips_per_dataset": 2,
        "num_frames": 4,
        "seed": 0,
        "output_dir": str(out_dir),
        "temporal_datasets": [],
        "model": "qwen3-omni",
    }
    base.update(overrides)
    return EvalConfig(**base)


def test_load_manifest_entries_samples_and_skips_blind(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)

    rows = [_entry("a1"), _entry("a2"), _entry("blind")]
    rows[2]["label"] = None
    with (manifest_dir / "tiny_train.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    cfg = _cfg(manifest_dir, tmp_path / "out", num_clips_per_dataset=2)
    entries = load_manifest_entries(cfg)
    assert len(entries) == 2
    assert all(e["label"] is not None for e in entries)
    assert {e["clip_id"] for e in entries} <= {"a1", "a2"}


def test_load_manifest_entries_inference_only_keeps_blind_labels(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)

    rows = [_entry("a1", "dfeval"), _entry("a2", "dfeval")]
    for row in rows:
        row["label"] = None
    with (manifest_dir / "dfeval.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    cfg = _cfg(
        manifest_dir,
        tmp_path / "out",
        dataset_sources={"dfeval": ["dfeval.jsonl"]},
        num_clips_per_dataset=2,
        inference_only_datasets=["dfeval"],
    )
    entries = load_manifest_entries(cfg)
    assert len(entries) == 2
    assert all(e["label"] is None for e in entries)


def test_load_manifest_entries_respects_seed(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)

    with (manifest_dir / "tiny_train.jsonl").open("w") as fh:
        for i in range(10):
            fh.write(json.dumps(_entry(f"c{i}")) + "\n")

    def make_cfg(seed: int) -> EvalConfig:
        return _cfg(manifest_dir, tmp_path / "out", num_clips_per_dataset=3, seed=seed)

    a = load_manifest_entries(make_cfg(42))
    b = load_manifest_entries(make_cfg(42))
    c = load_manifest_entries(make_cfg(7))
    assert [e["clip_id"] for e in a] == [e["clip_id"] for e in b]
    assert [e["clip_id"] for e in a] != [e["clip_id"] for e in c]


def test_load_manifest_entries_shards_are_disjoint_and_cover_sample(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)

    with (manifest_dir / "tiny_train.jsonl").open("w") as fh:
        for i in range(30):
            fh.write(json.dumps(_entry(f"c{i}")) + "\n")

    cfg = _cfg(manifest_dir, tmp_path / "out", num_clips_per_dataset=21, seed=42)

    all_entries = load_manifest_entries(cfg)
    shard_0 = load_manifest_entries(cfg, num_shards=3, shard_id=0)
    shard_1 = load_manifest_entries(cfg, num_shards=3, shard_id=1)
    shard_2 = load_manifest_entries(cfg, num_shards=3, shard_id=2)

    all_ids = {e["clip_id"] for e in all_entries}
    s0 = {e["clip_id"] for e in shard_0}
    s1 = {e["clip_id"] for e in shard_1}
    s2 = {e["clip_id"] for e in shard_2}

    assert s0.isdisjoint(s1)
    assert s1.isdisjoint(s2)
    assert s0.isdisjoint(s2)
    assert s0 | s1 | s2 == all_ids


def test_load_clip_resolves_paths_and_loads_media(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, str] = {}

    def _fake_load_video_frames(path: str) -> np.ndarray:
        calls["video_path"] = path
        return np.zeros((8, 4, 4, 3), dtype=np.uint8)

    def _fake_load_audio(path: str, sample_rate: int) -> np.ndarray:
        calls["audio_path"] = path
        return np.ones(100, dtype=np.float32)

    monkeypatch.setattr("deepfake_moe.eval.dataset.load_video_frames", _fake_load_video_frames)
    monkeypatch.setattr("deepfake_moe.eval.dataset.load_audio", _fake_load_audio)

    entry = {
        "clip_id": "c1",
        "dataset": "lavdf",
        "src_path": "test/000000.mp4",
        "wav_path": "audio/normalized/lavdf/test/000000.wav",
        "has_audio": True,
        "label": [1, 0],
        "fake_periods": [[1.0, 2.0]],
    }

    clip = load_clip(
        entry,
        num_frames=4,
        sample_rate=16_000,
        data_root="/data",
        audio_dir="/data/processed/audio/normalized",
        dataset_video_roots={"lavdf": "raw/LAV-DF/LAV-DF"},
    )

    assert clip is not None
    assert clip.clip_id == "c1"
    assert clip.dataset == "lavdf"
    assert clip.label_video == 1
    assert clip.label_audio == 0
    assert clip.frames.shape[0] == 4
    assert clip.has_audio is True
    assert calls["video_path"] == "/data/raw/LAV-DF/LAV-DF/test/000000.mp4"
    assert calls["audio_path"] == "/data/processed/audio/normalized/lavdf/test/000000.wav"


def test_load_clip_downgrades_has_audio_on_empty_waveform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "deepfake_moe.eval.dataset.load_video_frames",
        lambda path: np.zeros((8, 4, 4, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "deepfake_moe.eval.dataset.load_audio",
        lambda path, sample_rate: np.array([], dtype=np.float32),
    )

    entry = {
        "clip_id": "c1",
        "dataset": "lavdf",
        "src_path": "/abs/clip.mp4",
        "wav_path": "/abs/clip.wav",
        "has_audio": True,
        "label": [0, 0],
        "fake_periods": [],
    }

    clip = load_clip(entry, num_frames=4, sample_rate=16_000)
    assert clip is not None
    assert clip.has_audio is False
    assert clip.waveform is None


def test_load_clip_returns_none_on_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(path: str) -> np.ndarray:
        raise RuntimeError("decode failed")

    monkeypatch.setattr("deepfake_moe.eval.dataset.load_video_frames", _raise)

    entry = {
        "clip_id": "c1",
        "dataset": "lavdf",
        "src_path": "/abs/clip.mp4",
        "wav_path": None,
        "has_audio": False,
        "label": [0, 0],
        "fake_periods": [],
    }

    assert load_clip(entry, num_frames=4, sample_rate=16_000) is None
