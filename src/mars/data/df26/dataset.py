"""DF26 clip dataset."""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

from mars.data.base import BaseDataset
from mars.data.config import DataConfig
from mars.data.dataset import ClipDataset
from mars.data.df26.config import Df26Config
from mars.data.df26.manifest import load_records as load_df26_records
from mars.data.records import VideoRecord


class Df26Dataset(ClipDataset, BaseDataset):
    """DF26 clips. Sample.extra has method (real or a generator name)."""

    name = "df26"

    @classmethod
    def load_records(
        cls,
        cfg: DataConfig,
        *,
        is_main: bool = True,
        barrier: Callable[[], None] | None = None,
    ) -> list[VideoRecord]:
        """Read or build the DF26 manifest."""
        from mars.data.factory import corpus_mapping

        raw = corpus_mapping(cls.name, cfg)
        return load_df26_records(Df26Config.from_dict(raw), is_main=is_main, barrier=barrier)

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
        """Build one DF26 split from the run sampling config."""
        if records is None:
            records = cls.load_records(cfg, is_main=is_main, barrier=barrier)
        return cls(records, cfg, split=split)
