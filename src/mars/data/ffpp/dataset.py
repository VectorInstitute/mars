"""FaceForensics++ clip dataset."""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

from mars.data.base import BaseDataset
from mars.data.config import DataConfig
from mars.data.dataset import ClipDataset
from mars.data.ffpp.config import FfppConfig
from mars.data.ffpp.manifest import load_records as load_ffpp_records
from mars.data.records import VideoRecord


class FfppDataset(ClipDataset, BaseDataset):
    """FF++ clips. ``Sample.extra`` includes ``method``, ``compression``, ``group``."""

    name = "ffpp"

    @classmethod
    def load_records(
        cls,
        cfg: DataConfig,
        *,
        is_main: bool = True,
        barrier: Callable[[], None] | None = None,
    ) -> list[VideoRecord]:
        """Scan or read the FF++ identity-safe manifest."""
        from mars.data.factory import corpus_mapping

        raw = corpus_mapping(cls.name, cfg)
        return load_ffpp_records(FfppConfig.from_dict(raw), is_main=is_main, barrier=barrier)

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
        """Build one FF++ split from the run's sampling config."""
        if records is None:
            records = cls.load_records(cfg, is_main=is_main, barrier=barrier)
        return cls(records, cfg, split=split)
