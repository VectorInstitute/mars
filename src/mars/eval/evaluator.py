"""Scoring a detector: per-clip prediction, the test battery, standalone runs."""

from __future__ import annotations

import time
from typing import Any

import torch

from mars.data.loaders import build_loaders
from mars.eval.protocols import bullets, labels_and_scores, protocol_report, write_protocols
from mars.eval.summary import load_trainable_weights, summarise
from mars.eval.thresholds import SourceThreshold, load_source_threshold, source_threshold
from mars.eval.write import write_experiment_results
from mars.models.base import BaseDetector
from mars.train.config import RunConfig
from mars.utils.distributed import DistributedContext
from mars.utils.impact import Footprint
from mars.utils.logging import get_logger


AMP_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def frame_validity(n_valid: torch.Tensor, n_frames: int, device: torch.device) -> torch.Tensor:
    """(B, T) mask marking each clip's genuine frames, from the sampler's count."""
    return torch.arange(n_frames, device=device)[None, :] < n_valid.to(device)[:, None]


def _method_of(extra: dict[str, Any], index: int) -> str:
    values = extra.get("method")
    if not values or index >= len(values):
        return ""
    return "" if values[index] is None else str(values[index])


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: Any,
    ctx: DistributedContext,
    amp_dtype: torch.dtype = torch.bfloat16,
    dataset: str = "",
) -> dict[str, Any]:
    """Per-clip probabilities, gathered across ranks."""
    model.eval()
    if ctx.device.type == "cuda":
        torch.cuda.empty_cache()
    video_names = loader.dataset.video_ids
    rows: list[tuple[int, int, int, float, str]] = []

    log = get_logger(rank=0 if ctx.is_main else 1)
    total = len(loader)
    every = max(1, total // 20)
    started = time.perf_counter()
    log.info("eval %s: %d batches on this rank", dataset, total)

    use_amp = ctx.device.type == "cuda"
    for done, batch in enumerate(loader, start=1):
        frames = batch.frames.to(ctx.device, non_blocking=True)
        valid = frame_validity(batch.n_valid, frames.shape[1], ctx.device)
        with torch.autocast(ctx.device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(frames, valid=valid).logits
        probabilities = logits.float().softmax(dim=-1)[:, 1].cpu()
        for index, (video, clip, label, probability) in enumerate(
            zip(batch.video_index.tolist(), batch.clip_index.tolist(), batch.label.tolist(), probabilities.tolist())
        ):
            rows.append((video, clip, label, probability, _method_of(batch.extra, index)))
        if ctx.is_main and (done % every == 0 or done == total):
            rate = done / max(time.perf_counter() - started, 1e-9)
            log.info("eval %s: %d/%d batches (%.1f it/s)", dataset, done, total, rate)

    merged: dict[tuple[int, int], tuple[int, float, str]] = {}
    for shard in ctx.all_gather_object(rows):
        for video, clip, label, probability, method in shard:
            merged[(video, clip)] = (label, probability, method)

    ordered = sorted(merged)
    return {
        "video_ids": [video_names[video] for video, _ in ordered],
        "clip_indices": [clip for _, clip in ordered],
        "labels": [merged[key][0] for key in ordered],
        "scores": [merged[key][1] for key in ordered],
        "methods": [merged[key][2] for key in ordered],
    }


def evaluate(
    model: torch.nn.Module,
    loader: Any,
    ctx: DistributedContext,
    amp_dtype: torch.dtype = torch.bfloat16,
    dataset: str = "ffpp",
) -> dict[str, Any]:
    """Score one corpus and summarise it."""
    return summarise(predict(model, loader, ctx, amp_dtype, dataset=dataset), dataset)


def score_battery(
    model: torch.nn.Module,
    val_loader: Any,
    test_loaders: dict[str, Any],
    ctx: DistributedContext,
    cfg: RunConfig,
    footprint: Footprint,
    source: SourceThreshold | None = None,
) -> tuple[dict[str, Any], dict[str, Any], SourceThreshold]:
    """Score every test corpus at 0.5 and at the source threshold, and write it all.

    The source-validation pass and each corpus are measured as their own footprint
    stage, so a corpus dropped later takes its cost with it. A given source threshold
    (from an earlier run of the same checkpoint) skips the validation pass.

    Returns
    -------
    results : dict
        Summary per corpus.
    protocols : dict
        Threshold and ranking report per corpus.
    source : SourceThreshold
        Operating point fixed on source validation videos.
    """
    amp_dtype = AMP_DTYPES[cfg.train.amp_dtype]
    log = get_logger(rank=0 if ctx.is_main else 1)

    # The selected checkpoint's own source-validation videos fix the second
    # operating point; it is then applied unchanged to every test corpus.
    validation = None
    if source is None:
        footprint.start("source_val")
        validation = evaluate(model, val_loader, ctx, amp_dtype, dataset=cfg.data.val_corpus)
        footprint.stop("source_val")
        source = source_threshold(*labels_and_scores(validation, "video"))
    log.info(
        "source threshold %.6f (Youden J %.4f, %d maximisers) from %d validation videos",
        source.value,
        source.youden_j,
        source.n_maximisers,
        source.n_videos,
    )

    results: dict[str, Any] = {}
    for name, loader in test_loaders.items():
        footprint.start(f"eval/{name}")
        results[name] = evaluate(model, loader, ctx, amp_dtype, dataset=name)
        footprint.stop(f"eval/{name}")
    protocols = {name: protocol_report(result, source) for name, result in results.items()}
    if ctx.is_main:
        log.info("\n%s", bullets(protocols, source))
        write_experiment_results(results_dir=cfg.results_path, dataset_results=results)
        write_protocols(cfg.results_path, protocols, source, validation)
    return results, protocols, source


def run_split(
    detector: BaseDetector,
    ctx: DistributedContext,
    split: str = "test",
    checkpoint: str | None = None,
    dataset: str | None = None,
    source_threshold_file: str | None = None,
) -> dict[str, Any]:
    """Score one split with a checkpoint, the same way training scores it.

    A test run also scores source validation, so every corpus is reported at 0.5
    and at the checkpoint's own source threshold; with source_threshold_file (saved
    by an earlier run of the same checkpoint) that threshold is reused and the
    validation pass is skipped. The energy, carbon and water of each scored corpus
    go to its own impact.json, next to its predictions.
    """
    cfg = detector.cfg
    network = detector.build_network().to(ctx.device)
    if checkpoint:
        load_trainable_weights(network, checkpoint, ctx.device)
    footprint = Footprint(detector.base_model, detector.relation, ctx)

    if split == "test":
        if dataset and dataset not in cfg.data.test:
            raise ValueError(f"{dataset!r} is not in data.test={list(cfg.data.test)}")
        names = [dataset] if dataset else list(cfg.data.test)
        source = load_source_threshold(source_threshold_file) if source_threshold_file else None
        need = ("test",) if source is not None else ("val", "test")
        loaders, _ = build_loaders(cfg, ctx, need=need, test_names=names)
        results, _, _ = score_battery(network, loaders.get("val"), loaders["test"], ctx, cfg, footprint, source)
        footprint.write(cfg.results_path)
        return results

    if split not in ("train", "val"):
        raise ValueError(f"unknown split {split!r}")
    name = cfg.data.val_corpus if split == "val" else "+".join(cfg.data.train)
    loaders, _ = build_loaders(cfg, ctx, need=(split,))
    footprint.start(f"eval/{name}")
    results = {name: evaluate(network, loaders[split], ctx, AMP_DTYPES[cfg.train.amp_dtype], dataset=name)}
    footprint.stop(f"eval/{name}")
    if ctx.is_main:
        write_experiment_results(results_dir=cfg.results_path, dataset_results=results)
    footprint.write(cfg.results_path)
    return results
