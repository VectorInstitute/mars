"""AVID clip dataset."""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

from mars.data.avid.config import AvidConfig
from mars.data.avid.manifest import load_records as load_avid_records
from mars.data.base import BaseDataset
from mars.data.config import DataConfig
from mars.data.dataset import ClipDataset
from mars.data.records import VideoRecord


class AvidDataset(ClipDataset, BaseDataset):
    """AVID clips. Sample.extra has method (real or fake)."""

    name = "avid"

    @classmethod
    def load_records(
        cls,
        cfg: DataConfig,
        *,
        is_main: bool = True,
        barrier: Callable[[], None] | None = None,
    ) -> list[VideoRecord]:
        """Read or build the AVID manifest."""
        from mars.data.factory import corpus_mapping

        raw = corpus_mapping(cls.name, cfg)
        return load_avid_records(AvidConfig.from_dict(raw), is_main=is_main, barrier=barrier)

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
        """Build one AVID split from the run sampling config."""
        if records is None:
            records = cls.load_records(cfg, is_main=is_main, barrier=barrier)
        return cls(records, cfg, split=split)
