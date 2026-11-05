"""Clip dataset used by every corpus: one item is one ``n_frames`` segment."""

from __future__ import annotations

import zlib
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Self

import numpy as np
import torch
from torch.utils.data import Dataset

from mars.data.base import BaseDataset, Sample
from mars.data.config import DataConfig
from mars.data.perturb import perturb
from mars.data.records import VideoRecord
from mars.data.sampling import Clip, clips_for_video
from mars.data.transforms import preprocess
from mars.data.video import read_frames
from mars.utils.logging import get_logger


def clip_seed(clip: Clip) -> int:
    """Seed of a clip's random perturbation, fixed by its video and position."""
    return zlib.crc32(f"{clip.video_id}#{clip.clip_index}".encode())


def build_clips(records: list[VideoRecord], cfg: DataConfig) -> list[Clip]:
    """Enumerate clips for every record under ``cfg`` sampling rules."""
    clips: list[Clip] = []
    for record in records:
        clips.extend(
            clips_for_video(
                video_id=record.video_id,
                n_source_frames=record.n_frames,
                n_frames=cfg.n_frames,
                frame_stride=cfg.frame_stride,
                clip_stride=cfg.clip_stride,
                padding=cfg.padding,
            )
        )
    return clips


class ClipDataset(Dataset):
    """Shared clip reader. Corpus classes wrap this and set ``name``.

    Yields :class:`Sample` with required tensors. ``Sample.extra`` is copied
    from :class:`VideoRecord.extra`.
    """

    def __init__(
        self,
        records: list[VideoRecord],
        cfg: DataConfig,
        split: str | None = None,
    ) -> None:
        if split is not None:
            records = [r for r in records if r.split == split]
        self.cfg = cfg
        self.records = {r.video_id: r for r in records}
        self.clips = build_clips(records, cfg)
        self._video_ids = sorted(self.records)
        self._video_index = {v: i for i, v in enumerate(self._video_ids)}

    def __len__(self) -> int:
        """Return the number of clips."""
        return len(self.clips)

    @property
    def labels(self) -> list[int]:
        """Clip-level labels aligned with ``self.clips``."""
        return [self.records[c.video_id].label for c in self.clips]

    @property
    def video_ids(self) -> list[str]:
        """Video ids indexed by ``Sample.video_index``."""
        return self._video_ids

    def __getitem__(self, index: int) -> Sample:
        """Return one clip sample.

        Falls back to a blank clip if the file fails to decode at read time
        (e.g. transient I/O error) rather than crashing the whole run - one
        bad file should not take down a multi-hour distributed job.
        """
        clip = self.clips[index]
        record = self.records[clip.video_id]
        try:
            frames = read_frames(record.path, list(clip.frame_indices))
        except RuntimeError:
            get_logger().warning(
                "failed to decode clip %d of %s (%s); using a blank clip",
                clip.clip_index,
                clip.video_id,
                record.path,
            )
            frames = np.zeros((len(clip.frame_indices), self.cfg.image_size, self.cfg.image_size, 3), dtype=np.uint8)
        if self.cfg.perturb != "none":
            frames = perturb(frames, self.cfg.perturb, self.cfg.perturb_level, seed=clip_seed(clip))
        return Sample(
            frames=preprocess(frames, self.cfg.image_size),
            label=torch.tensor(record.label, dtype=torch.long),
            n_valid=torch.tensor(clip.n_valid, dtype=torch.long),
            video_index=torch.tensor(self._video_index[clip.video_id], dtype=torch.long),
            clip_index=torch.tensor(clip.clip_index, dtype=torch.long),
            extra=dict(record.extra),
        )


class ConcatClipDataset(BaseDataset):
    """Train-time concat of two or more clip datasets."""

    def __init__(self, datasets: Sequence[BaseDataset]) -> None:
        """Stack clip datasets in order; prefix video ids with each corpus name."""
        if len(datasets) < 2:
            raise ValueError("ConcatClipDataset needs at least two datasets")
        self.datasets = list(datasets)
        self.name = "+".join(dataset.name for dataset in self.datasets)
        clip_starts = [0]
        video_starts = [0]
        video_ids: list[str] = []
        for dataset in self.datasets:
            clip_starts.append(clip_starts[-1] + len(dataset))
            prefix = f"{dataset.name}/"
            for video_id in dataset.video_ids:
                video_ids.append(video_id if video_id.startswith(prefix) else prefix + video_id)
            video_starts.append(len(video_ids))
        self._clip_starts = clip_starts
        self._video_starts = video_starts
        self._video_ids = video_ids

    @classmethod
    def from_config(
        cls,
        cfg: DataConfig,
        split: str | None,
        *,
        is_main: bool = True,
        barrier: Callable[[], None] | None = None,
        records: list[VideoRecord] | None = None,
    ) -> Self:
        """Raise; concatenate train corpora through the factory instead."""
        del cls, cfg, split, is_main, barrier, records
        raise TypeError("ConcatClipDataset is built by the factory")

    @classmethod
    def load_records(
        cls,
        cfg: DataConfig,
        *,
        is_main: bool = True,
        barrier: Callable[[], None] | None = None,
    ) -> list[VideoRecord]:
        """Raise; concatenate train corpora through the factory instead."""
        del cls, cfg, is_main, barrier
        raise TypeError("ConcatClipDataset is built by the factory")

    @property
    def labels(self) -> list[int]:
        """Clip-level labels aligned with concat indices."""
        labels: list[int] = []
        for dataset in self.datasets:
            labels.extend(dataset.labels)
        return labels

    @property
    def video_ids(self) -> list[str]:
        """Video ids indexed by Sample.video_index."""
        return self._video_ids

    def __len__(self) -> int:
        """Return the number of clips."""
        return self._clip_starts[-1]

    def __getitem__(self, index: int) -> Sample:
        """Return one clip sample from the owning corpus."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        for dataset_index, dataset in enumerate(self.datasets):
            start = self._clip_starts[dataset_index]
            stop = self._clip_starts[dataset_index + 1]
            if start <= index < stop:
                sample = dataset[index - start]
                video_index = self._video_starts[dataset_index] + int(sample.video_index)
                return replace(sample, video_index=torch.tensor(video_index, dtype=torch.long))
        raise IndexError(index)


def collate(batch: list[Sample]) -> Sample:
    """Stack required tensors; align extras as lists (None if a row omits a key)."""
    keys = sorted({key for sample in batch for key in sample.extra})
    extra = {key: [sample.extra.get(key) for sample in batch] for key in keys}
    return Sample(
        frames=torch.stack([sample.frames for sample in batch]),
        label=torch.stack([sample.label for sample in batch]),
        n_valid=torch.stack([sample.n_valid for sample in batch]),
        video_index=torch.stack([sample.video_index for sample in batch]),
        clip_index=torch.stack([sample.clip_index for sample in batch]),
        extra=extra,
    )
