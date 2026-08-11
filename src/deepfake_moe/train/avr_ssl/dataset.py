"""Real-only, has-audio manifest dataset for AVR Stage 1 self-supervised training."""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from deepfake_moe.train.common.splits import read_manifest_pool, uniform_sample
from deepfake_moe.utils.logging_utils import get_logger
from deepfake_moe.utils.media_io import load_audio, load_video_frames, video_native_fps
from deepfake_moe.utils.paths import resolve_audio_path, resolve_video_path
from deepfake_moe.utils.video_segment import sample_frame_indices


logger = get_logger(__name__)


def filter_real_only_with_audio(pool: list[dict]) -> list[dict]:
    """Keep real-labeled (``label == [0, 0]``) entries that have an audio stream.

    Applied as two separate, separately-logged steps rather than one
    combined filter, so each drop's contribution stays visible. Video-only
    real clips can't supply a contrastive pair or support V2A imputation, so
    Stage 1 drops them entirely rather than special-casing them.

    Examples
    --------
    >>> pool = [
    ...     {"label": [0, 0], "has_audio": True},
    ...     {"label": [0, 0], "has_audio": False},
    ...     {"label": [1, 0], "has_audio": True},
    ... ]
    >>> len(filter_real_only_with_audio(pool))
    1
    """
    real = [entry for entry in pool if list(entry.get("label") or [1, 1]) == [0, 0]]
    with_audio = [entry for entry in real if entry.get("has_audio", False)]
    logger.info(
        "real-only filter: pool=%d -> real=%d -> real_with_audio=%d (dropped %d, no audio)",
        len(pool),
        len(real),
        len(with_audio),
        len(real) - len(with_audio),
    )
    return with_audio


def load_real_only_entries(
    manifest_dir: str,
    dataset_sources: dict[str, list[str]],
    samples_per_dataset: int,
    *,
    seed: int,
) -> list[dict]:
    """Load, filter to real-only + has-audio, then sample per dataset.

    Filters before sampling, not after: ``samples_per_dataset`` caps each
    dataset's REAL-only pool, not its raw (real + fake) pool. Real-only pool
    sizes are already very uneven across datasets, so sampling from the raw
    pool first would let that imbalance leak into the cap's meaning. Always
    uniform, never stratified: once the pool is real-only there is only one
    class left to stratify by.
    """
    all_entries: list[dict] = []
    for dataset, filenames in dataset_sources.items():
        pool = read_manifest_pool(manifest_dir, filenames)
        real_with_audio = filter_real_only_with_audio(pool)
        if not real_with_audio:
            logger.warning("No usable real, has-audio entries for dataset=%s", dataset)
            continue
        sampled = uniform_sample(real_with_audio, samples_per_dataset, seed=seed)
        logger.info("dataset=%-15s  real_with_audio=%6d  sampled=%d", dataset, len(real_with_audio), len(sampled))
        all_entries.extend(sampled)
    logger.info("Total avr_ssl entries: %d", len(all_entries))
    return all_entries


class AVRSSLDataset(Dataset):
    """Decodes real, has-audio manifest entries to frames + waveform.

    Unlike ``train/common/data.py``'s ``SFTDataset``, this returns raw
    decoded media, not a ``ClipData``/prompt pair — Stage 1 trains on
    encoder-output-level reconstruction targets, not prompt/target text, and
    the frozen encoders + processor live on GPU in the trainer, not here.

    Video frames are uniformly subsampled to ``num_frames`` (same mechanism
    and default as the rest of this repo's training configs — see
    ``sample_frame_indices``), which bounds the per-clip video token count
    regardless of clip duration, rather than dropping long clips outright.
    Audio is kept full-length (no comparable token-count blowup — 13 Hz mel
    tokens, no per-frame spatial multiplier). Since ``sample_frame_indices``
    picks frames evenly across the WHOLE clip rather than truncating a
    prefix, both streams still cover the same full time range — just at
    different token densities — so no audio/video duration mismatch is
    introduced. The processor needs the subsampled stream's own effective
    frame rate (not the source video's native fps) to compute correct token
    timestamps; see ``effective_fps`` below.
    """

    def __init__(
        self,
        entries: list[dict],
        *,
        num_frames: int,
        audio_sample_rate: int,
        data_root: str = "",
        audio_dir: str = "",
        dataset_video_roots: dict[str, str] | None = None,
    ) -> None:
        self.entries = entries
        self.num_frames = num_frames
        self.audio_sample_rate = audio_sample_rate
        self.data_root = data_root
        self.audio_dir = audio_dir
        self.dataset_video_roots = dataset_video_roots or {}

    def __len__(self) -> int:
        """Return the configured entry count."""
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Decode one entry's subsampled frames, waveform, and effective fps."""
        entry = self.entries[index]
        clip_id = entry.get("clip_id", index)
        dataset = entry.get("dataset", "")
        video_root = self.dataset_video_roots.get(dataset, "")
        src_path = resolve_video_path(entry.get("src_path", ""), data_root=self.data_root, video_root=video_root)
        wav_path = resolve_audio_path(entry.get("wav_path"), audio_dir=self.audio_dir)
        if wav_path is None:
            raise RuntimeError(f"Entry {clip_id!r} has has_audio=True but no wav_path")

        all_frames = load_video_frames(src_path)
        waveform = load_audio(wav_path, sample_rate=self.audio_sample_rate)
        if len(all_frames) == 0 or len(waveform) == 0:
            raise RuntimeError(f"Failed to decode training clip {clip_id!r}")

        native_fps = video_native_fps(src_path)
        indices = sample_frame_indices(len(all_frames), self.num_frames)
        frames = all_frames[indices]
        clip_duration = len(all_frames) / native_fps
        effective_fps = len(indices) / clip_duration

        return {
            "clip_id": clip_id,
            "dataset": dataset,
            "frames": frames,
            "waveform": waveform,
            "fps": effective_fps,
        }
