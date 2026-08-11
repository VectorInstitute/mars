"""Eval manifest loading and per-clip media decode."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np
import yaml

from deepfake_moe.models.base import ClipData
from deepfake_moe.utils.env import expand_env
from deepfake_moe.utils.logging_utils import get_logger
from deepfake_moe.utils.media_io import load_audio, load_video_frames
from deepfake_moe.utils.paths import default_audio_dir, resolve_audio_path, resolve_video_path
from deepfake_moe.utils.video_segment import sample_frame_indices


logger = get_logger(__name__)


@dataclass
class EvalConfig:
    """Typed view over ``configs/eval.yaml``.

    Kept separate from the model ``Config`` so eval settings do not affect
    model-config validation. ``model`` names an entry in ``configs/models.yaml``;
    results are written to ``<output_dir>/<experiment_name>/<model>/``.
    """

    manifest_dir: str
    dataset_sources: dict[str, list[str]]
    num_clips_per_dataset: int
    num_frames: int
    seed: int
    output_dir: str
    temporal_datasets: list[str]
    model: str
    experiment_name: str = "default"
    data_root: str = ""
    dataset_video_roots: dict[str, str] = field(default_factory=dict)
    inference_only_datasets: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        """Load eval config from YAML."""
        data = expand_env(yaml.safe_load(Path(path).read_text()) or {})
        known = {f.name for f in dc_fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown EvalConfig keys: {sorted(unknown)}")
        return cls(**data)

    def resolved_audio_dir(self) -> str:
        """Return ``{data_root}/processed/audio/normalized``."""
        return default_audio_dir(self.data_root)


def load_manifest_entries(
    eval_cfg: EvalConfig,
    *,
    num_shards: int = 1,
    shard_id: int = 0,
) -> list[dict]:
    """Load and sample manifest entries for every configured dataset.

    Reads ``<manifest_dir>/<filename>`` directly for each configured source.
    Rows with ``label=null`` are skipped unless their dataset is listed in
    ``inference_only_datasets`` (e.g. ``dfeval``, a blind heldout set that is
    run for coverage/parse-rate only, never scored).

    Returns
    -------
    list[dict]
        Flat list of sampled manifest dicts for ``load_clip``.
    """
    _validate_shard_args(num_shards, shard_id)
    rng = np.random.default_rng(eval_cfg.seed)
    manifest_dir = Path(eval_cfg.manifest_dir)
    all_entries: list[dict] = []

    for dataset, filenames in eval_cfg.dataset_sources.items():
        pool: list[dict] = []
        inference_only = dataset in eval_cfg.inference_only_datasets

        for fname in filenames:
            fpath = manifest_dir / fname
            if not fpath.exists():
                logger.warning("manifest not found, skipping: %s", fpath)
                continue
            with open(fpath) as fh:
                for line in fh:
                    text = line.strip()
                    if not text:
                        continue
                    entry = json.loads(text)
                    if entry.get("label") is None and not inference_only:
                        continue
                    pool.append(entry)

        if not pool:
            logger.warning("No usable entries for dataset=%s", dataset)
            continue

        n = min(eval_cfg.num_clips_per_dataset, len(pool))
        indices = rng.choice(len(pool), size=n, replace=False)
        sampled = [pool[i] for i in sorted(indices)]
        sharded = [entry for entry in sampled if _belongs_to_shard(entry, num_shards=num_shards, shard_id=shard_id)]
        logger.info(
            "dataset=%-15s  pool=%6d  sampled=%d  shard=%d/%d  kept=%d",
            dataset,
            len(pool),
            n,
            shard_id,
            num_shards,
            len(sharded),
        )
        all_entries.extend(sharded)

    logger.info("Total clips queued (shard %d/%d): %d", shard_id, num_shards, len(all_entries))
    return all_entries


def _validate_shard_args(num_shards: int, shard_id: int) -> None:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards - 1}], got {shard_id}")


def _shard_key(entry: dict) -> str:
    dataset = entry.get("dataset", "")
    clip_id = entry.get("clip_id") or entry.get("src_path", "")
    return f"{dataset}:{clip_id}"


def _belongs_to_shard(entry: dict, *, num_shards: int, shard_id: int) -> bool:
    key = _shard_key(entry)
    digest = hashlib.md5(key.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest, 16) % num_shards == shard_id


def load_clip(
    entry: dict,
    *,
    num_frames: int,
    sample_rate: int,
    data_root: str = "",
    audio_dir: str = "",
    dataset_video_roots: dict[str, str] | None = None,
) -> ClipData | None:
    """Load video frames and audio for one manifest entry.

    Returns ``None`` when the source file is missing or unreadable.

    Parameters
    ----------
    entry :
        One manifest JSON object.
    num_frames :
        Target number of frames after uniform subsampling.
    sample_rate :
        Audio sample rate for ``wav_path`` loading.
    data_root :
        Common parent of the raw dataset folders; combined with
        ``dataset_video_roots[entry["dataset"]]`` to resolve ``src_path``.
    audio_dir :
        Root for normalized audio; combined with ``wav_path`` to resolve the
        on-disk audio file.
    dataset_video_roots :
        Maps a manifest ``dataset`` name onto its raw-data subfolder (manifest
        dataset names don't match the on-disk folder names, e.g.
        ``lavdf`` -> ``raw/LAV-DF/LAV-DF``).

    Returns
    -------
    ClipData | None
        Decoded clip, or ``None`` on load failure.
    """
    dataset = entry.get("dataset", "")
    raw_src_path: str = entry.get("src_path", "")
    raw_wav_path: str | None = entry.get("wav_path")
    has_audio: bool = bool(entry.get("has_audio", False))
    label: list[int] = entry.get("label") or [0, 0]
    fake_periods: list[list[float]] = entry.get("fake_periods") or []

    video_root = (dataset_video_roots or {}).get(dataset, "")
    src_path = resolve_video_path(raw_src_path, data_root=data_root, video_root=video_root)
    wav_path = resolve_audio_path(raw_wav_path, audio_dir=audio_dir)

    try:
        all_frames = load_video_frames(src_path)
    except Exception as exc:
        logger.error(
            "video load failed  clip=%s  path=%s  error=%s",
            entry.get("clip_id"),
            src_path,
            exc,
        )
        return None

    if len(all_frames) == 0:
        logger.warning("empty video clip=%s  path=%s", entry.get("clip_id"), src_path)
        return None

    indices = sample_frame_indices(len(all_frames), num_frames)
    frames = all_frames[indices]

    waveform: np.ndarray | None = None
    if has_audio and wav_path:
        try:
            waveform = load_audio(wav_path, sample_rate=sample_rate)
            if len(waveform) == 0:
                logger.warning("empty waveform  clip=%s", entry.get("clip_id"))
                waveform = None
                has_audio = False
        except Exception as exc:
            logger.error(
                "audio load failed  clip=%s  path=%s  error=%s",
                entry.get("clip_id"),
                wav_path,
                exc,
            )
            waveform = None
            has_audio = False

    return ClipData(
        clip_id=entry.get("clip_id", ""),
        dataset=dataset,
        has_audio=has_audio,
        frames=frames,
        waveform=waveform,
        label_video=int(label[0]),
        label_audio=int(label[1]),
        fake_periods=fake_periods,
        video_path=src_path,
        audio_path=wav_path if has_audio and wav_path else None,
    )
