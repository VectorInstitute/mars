"""Dataset contract: required sample fields plus optional extras."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Self

import torch
from torch.utils.data import Dataset

from mars.data.config import DataConfig


@dataclass
class Sample:
    """One clip, or a collated batch of clips.

    Required for every corpus. ``extra`` holds optional per-clip fields
    (FF++ ``method``, compression, ...). Missing keys mean the trainer and
    evaluator skip that breakdown.
    """

    frames: torch.Tensor
    label: torch.Tensor
    n_valid: torch.Tensor
    video_index: torch.Tensor
    clip_index: torch.Tensor
    extra: dict[str, Any] = field(default_factory=dict)


class BaseDataset(Dataset, ABC):
    """One registered corpus. Factory dispatches on ``name``.

    ``__getitem__`` must return a :class:`Sample`. Subclasses may put extra
    keys on ``Sample.extra``; they must not drop the required tensors.
    """

    name: str

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        cfg: DataConfig,
        split: str | None,
        *,
        is_main: bool = True,
        barrier: Callable[[], None] | None = None,
        records: list[Any] | None = None,
    ) -> Self:
        """Build the split requested by the run YAML."""

    @classmethod
    @abstractmethod
    def load_records(
        cls,
        cfg: DataConfig,
        *,
        is_main: bool = True,
        barrier: Callable[[], None] | None = None,
    ) -> list[Any]:
        """Return every video row for this corpus (all splits)."""

    @property
    @abstractmethod
    def labels(self) -> list[int]:
        """Clip-level labels aligned with dataset indices."""

    @property
    @abstractmethod
    def video_ids(self) -> list[str]:
        """Video ids indexed by ``Sample.video_index``."""

    @abstractmethod
    def __getitem__(self, index: int) -> Sample:
        """Return one clip sample."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of clips."""
