"""GenVideo clip dataset."""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

from mars.data.base import BaseDataset
from mars.data.config import DataConfig
from mars.data.dataset import ClipDataset
from mars.data.genvideo.config import GenvideoConfig
from mars.data.genvideo.manifest import load_records as load_genvideo_records
from mars.data.records import VideoRecord


class GenvideoDataset(ClipDataset, BaseDataset):
    """GenVideo clips. Sample.extra has method (real or a generator name)."""

    name = "genvideo"

    @classmethod
    def load_records(
        cls,
        cfg: DataConfig,
        *,
        is_main: bool = True,
        barrier: Callable[[], None] | None = None,
    ) -> list[VideoRecord]:
        """Read or build the GenVideo manifest."""
        from mars.data.factory import corpus_mapping

        raw = corpus_mapping(cls.name, cfg)
        return load_genvideo_records(GenvideoConfig.from_dict(raw), is_main=is_main, barrier=barrier)

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
        """Build one GenVideo split from the run sampling config."""
        if records is None:
            records = cls.load_records(cfg, is_main=is_main, barrier=barrier)
        return cls(records, cfg, split=split)
