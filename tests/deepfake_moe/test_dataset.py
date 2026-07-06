"""Tests for manifest loading."""

import json
from pathlib import Path

from deepfake_moe.eval.dataset import EvalConfig, load_manifest_entries


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


def test_load_manifest_entries_samples_and_skips_blind(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifests"
    by_dataset = manifest_dir / "by_dataset"
    by_dataset.mkdir(parents=True)

    rows = [_entry("a1"), _entry("a2"), _entry("blind")]
    rows[2]["label"] = None
    with (by_dataset / "tiny_train.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    cfg = EvalConfig(
        manifest_dir=str(manifest_dir),
        dataset_sources={"tiny": ["tiny_train.jsonl"]},
        num_clips_per_dataset=2,
        num_frames=4,
        seed=0,
        output_dir=str(tmp_path / "out"),
        temporal_datasets=[],
        model="qwen3-omni",
    )

    entries = load_manifest_entries(cfg)
    assert len(entries) == 2
    assert all(e["label"] is not None for e in entries)
    assert {e["clip_id"] for e in entries} <= {"a1", "a2"}


def test_load_manifest_entries_respects_seed(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifests"
    by_dataset = manifest_dir / "by_dataset"
    by_dataset.mkdir(parents=True)

    with (by_dataset / "tiny_train.jsonl").open("w") as fh:
        for i in range(10):
            fh.write(json.dumps(_entry(f"c{i}")) + "\n")

    def make_cfg(seed: int) -> EvalConfig:
        return EvalConfig(
            manifest_dir=str(manifest_dir),
            dataset_sources={"tiny": ["tiny_train.jsonl"]},
            num_clips_per_dataset=3,
            num_frames=4,
            seed=seed,
            output_dir=str(tmp_path / "out"),
            temporal_datasets=[],
            model="qwen3-omni",
        )

    a = load_manifest_entries(make_cfg(42))
    b = load_manifest_entries(make_cfg(42))
    c = load_manifest_entries(make_cfg(7))
    assert [e["clip_id"] for e in a] == [e["clip_id"] for e in b]
    assert [e["clip_id"] for e in a] != [e["clip_id"] for e in c]


def test_load_manifest_entries_shards_are_disjoint_and_cover_sample(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifests"
    by_dataset = manifest_dir / "by_dataset"
    by_dataset.mkdir(parents=True)

    with (by_dataset / "tiny_train.jsonl").open("w") as fh:
        for i in range(30):
            fh.write(json.dumps(_entry(f"c{i}")) + "\n")

    cfg = EvalConfig(
        manifest_dir=str(manifest_dir),
        dataset_sources={"tiny": ["tiny_train.jsonl"]},
        num_clips_per_dataset=21,
        num_frames=4,
        seed=42,
        output_dir=str(tmp_path / "out"),
        temporal_datasets=[],
        model="qwen3-omni",
    )

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
