"""Dispatch a dataset name to its class, same pattern as ``models.factory``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from mars.config import load_corpus_yaml, merge
from mars.data.base import BaseDataset
from mars.data.config import DataConfig
from mars.data.dataset import ConcatClipDataset
from mars.data.records import VideoRecord


def corpus_mapping(name: str, cfg: DataConfig) -> dict[str, Any]:
    """Corpus YAML for ``name``, with ``cfg.overrides`` applied on top."""
    return merge(load_corpus_yaml(name), cfg.overrides)


def require_role(name: str, split: str, raw: dict[str, Any]) -> None:
    """Raise if ``split`` is not in the corpus ``roles`` list."""
    roles = list(raw.get("roles") or [])
    if split not in roles:
        raise ValueError(f"cannot use {name!r} as {split}: roles={roles}")


def dataset_class(name: str, raw: dict[str, Any] | None = None) -> type[BaseDataset]:
    """Return the dataset class registered under ``name``."""
    slug = str((raw or {}).get("dataset") or name)
    if slug == "ffpp":
        from mars.data.ffpp.dataset import FfppDataset

        cls: type[BaseDataset] = FfppDataset
    elif slug == "celebdf":
        from mars.data.celebdf.dataset import CelebdfDataset

        cls = CelebdfDataset
    elif slug == "dftimit":
        from mars.data.dftimit.dataset import DftimitDataset

        cls = DftimitDataset
    elif slug == "avid":
        from mars.data.avid.dataset import AvidDataset

        cls = AvidDataset
    elif slug == "uadfv":
        from mars.data.uadfv.dataset import UadfvDataset

        cls = UadfvDataset
    elif slug == "genvideo":
        from mars.data.genvideo.dataset import GenvideoDataset

        cls = GenvideoDataset
    elif slug == "vifbench":
        from mars.data.vifbench.dataset import VifbenchDataset

        cls = VifbenchDataset
    elif slug == "df26":
        from mars.data.df26.dataset import Df26Dataset

        cls = Df26Dataset
    elif slug == "hififace_paired":
        from mars.data.hififace.dataset import HififacePairedDataset

        cls = HififacePairedDataset
    elif slug == "hififace_onlyfake":
        from mars.data.hififace.dataset import HififaceOnlyfakeDataset

        cls = HififaceOnlyfakeDataset
    elif slug == "gtav":
        from mars.data.gtav.dataset import GtavDataset

        cls = GtavDataset
    else:
        raise ValueError(f"unknown dataset {slug!r}")
    return cls


def load_records(
    name: str,
    cfg: DataConfig,
    *,
    is_main: bool = True,
    barrier: Callable[[], None] | None = None,
) -> list[VideoRecord]:
    """Load video rows for corpus ``name`` (all splits)."""
    raw = corpus_mapping(name, cfg)
    return dataset_class(name, raw).load_records(cfg, is_main=is_main, barrier=barrier)


def build_dataset(
    name: str,
    cfg: DataConfig,
    split: str,
    *,
    records: list[VideoRecord] | None = None,
    is_main: bool = True,
    barrier: Callable[[], None] | None = None,
) -> BaseDataset:
    """Construct the dataset class for ``name`` and ``split``."""
    raw = corpus_mapping(name, cfg)
    require_role(name, split, raw)
    cls = dataset_class(name, raw)
    return cls.from_config(
        cfg,
        split,
        is_main=is_main,
        barrier=barrier,
        records=records,
    )


def build_split_datasets(
    cfg: DataConfig,
    *,
    need: Sequence[str] = ("train", "val", "test"),
    test_names: Sequence[str] | None = None,
    is_main: bool = True,
    barrier: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Build the clip datasets this run asked for.

    Returns
    -------
    dict
        ``train`` and ``val`` map to a :class:`BaseDataset`. ``test`` maps to
        ``{corpus: BaseDataset}``.
    """
    cache: dict[str, list[VideoRecord]] = {}

    def records(name: str) -> list[VideoRecord]:
        if name not in cache:
            cache[name] = load_records(name, cfg, is_main=is_main, barrier=barrier)
        return cache[name]

    def one(name: str, split: str) -> BaseDataset:
        raw = corpus_mapping(name, cfg)
        require_role(name, split, raw)
        return build_dataset(
            name,
            cfg,
            split,
            records=records(name),
            is_main=is_main,
            barrier=barrier,
        )

    out: dict[str, Any] = {}
    if "train" in need:
        names = list(cfg.train)
        if not names:
            raise ValueError("data.train is required")
        parts = [one(name, "train") for name in names]
        out["train"] = parts[0] if len(parts) == 1 else ConcatClipDataset(parts)
    if "val" in need:
        out["val"] = one(cfg.val_corpus, "val")
    if "test" in need:
        names = list(test_names if test_names is not None else cfg.test)
        if not names:
            raise ValueError("data.test is required (e.g. test: [ffpp])")
        out["test"] = {name: one(name, "test") for name in names}
    return out
