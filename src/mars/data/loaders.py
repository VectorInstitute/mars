"""Data loaders for the train, validation and test splits."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from mars.data.base import BaseDataset
from mars.data.dataset import collate
from mars.data.factory import build_split_datasets
from mars.train.config import RunConfig
from mars.utils.distributed import DistributedContext


def _make_loader(dataset: BaseDataset, *, is_train: bool, cfg: RunConfig, ctx: DistributedContext) -> DataLoader:
    """Wrap one clip dataset with the run's batching and sampler settings."""
    sampler: Sampler[int] | None = None
    if ctx.enabled:
        sampler = DistributedSampler(dataset, shuffle=is_train, drop_last=False)
    elif is_train and cfg.train.class_balance == "sampler":
        labels = torch.tensor(dataset.labels)
        counts = torch.bincount(labels, minlength=cfg.model.n_classes).float()
        weights = (1.0 / counts.clamp(min=1))[labels]
        sampler = WeightedRandomSampler(weights.tolist(), len(weights), replacement=True)
    return DataLoader(
        dataset,
        batch_size=(
            cfg.train.micro_batch_size if is_train else cfg.train.eval_batch_size or cfg.train.micro_batch_size
        ),
        shuffle=is_train and sampler is None,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=collate,
        pin_memory=ctx.device.type == "cuda",
        drop_last=False,
        persistent_workers=cfg.data.num_workers > 0,
    )


def build_loaders(
    cfg: RunConfig,
    ctx: DistributedContext,
    *,
    need: tuple[str, ...] = ("train", "val", "test"),
    test_names: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Wrap the requested corpora in loaders, keyed by split; test by corpus."""
    datasets = build_split_datasets(
        cfg.data, need=need, test_names=test_names, is_main=ctx.is_main, barrier=ctx.barrier
    )
    loaders: dict[str, Any] = {}
    if "train" in datasets:
        loaders["train"] = _make_loader(datasets["train"], is_train=True, cfg=cfg, ctx=ctx)
    if "val" in datasets:
        loaders["val"] = _make_loader(datasets["val"], is_train=False, cfg=cfg, ctx=ctx)
    if "test" in datasets:
        loaders["test"] = {
            name: _make_loader(dataset, is_train=False, cfg=cfg, ctx=ctx) for name, dataset in datasets["test"].items()
        }
    return loaders, datasets


def class_weights_for(dataset: BaseDataset, cfg: RunConfig) -> torch.Tensor | None:
    """Inverse-frequency class weights when class_balance is weighted_ce."""
    if cfg.train.class_balance != "weighted_ce":
        return None
    counts = torch.bincount(torch.tensor(dataset.labels), minlength=cfg.model.n_classes).float()
    return counts.sum() / (counts.clamp(min=1) * cfg.model.n_classes)
