"""Training loop shared by every detector: AdamW, step decay, bf16, DDP, selection."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from mars.config import to_dict
from mars.data.loaders import build_loaders, class_weights_for
from mars.eval.evaluator import AMP_DTYPES, evaluate, frame_validity, score_battery
from mars.eval.protocols import flat
from mars.eval.summary import load_trainable_weights
from mars.models.base import BaseDetector
from mars.train.config import RunConfig
from mars.train.init import reseed_linear
from mars.utils.distributed import DistributedContext
from mars.utils.impact import Footprint
from mars.utils.logging import RunLogger
from mars.utils.seed import seed_everything


Optimisation = tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]


def save_checkpoint(
    path: Path,
    network: nn.Module,
    objective: nn.Module,
    optimisation: Optimisation,
    epoch: int,
    best: float,
    cfg: RunConfig,
) -> None:
    """Write trainable weights, loss and optimizer state; skip the frozen encoder."""
    optimizer, scheduler = optimisation
    tuned = {name for name, p in network.named_parameters() if p.requires_grad}
    weights = {k: v for k, v in network.state_dict().items() if not k.startswith("encoder.") or k in tuned}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best": best,
            "select_on": cfg.train.select_on,
            "config": to_dict(cfg),
            "model": weights,
            "objective": objective.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        path,
    )


def sync_buffers_(objective: nn.Module, ctx: DistributedContext) -> None:
    """Average the loss's running state over ranks so every rank holds the same."""
    if not ctx.enabled:
        return
    with torch.no_grad():
        for buffer in objective.buffers():
            if buffer.is_floating_point() and buffer.numel():
                dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
                buffer.div_(ctx.world_size)


def _optimisation(cfg: RunConfig, parameters: list[nn.Parameter]) -> Optimisation:
    """AdamW over the trainable parameters, with a step decay."""
    optimizer = torch.optim.AdamW(
        parameters,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=(cfg.train.betas[0], cfg.train.betas[1]),
        eps=cfg.train.eps,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.train.decay_steps, cfg.train.decay_rate)
    return optimizer, scheduler


def _select_and_save(
    model: nn.Module,
    network: nn.Module,
    objective: nn.Module,
    val_loader: Any,
    ctx: DistributedContext,
    cfg: RunConfig,
    logger: RunLogger,
    optimisation: Optimisation,
    epoch: int,
    step: int,
    best: float,
) -> float:
    """Score the validation split, then write best.pt and last.pt."""
    results = evaluate(model, val_loader, ctx, AMP_DTYPES[cfg.train.amp_dtype], dataset=cfg.data.val_corpus)
    video = results["video"]
    logger.log_metrics(
        {**results["clip"].as_dict("val/clip/"), **video.as_dict("val/video/"), "epoch": epoch},
        step=step,
        table="val",
    )
    logger.info(
        "epoch %d | val video roc_auc %.4f pr_auc %.4f acc %.4f | clip roc_auc %.4f | selecting on %s",
        epoch,
        video.roc_auc or float("nan"),
        video.pr_auc or float("nan"),
        video.accuracy,
        results["clip"].roc_auc or float("nan"),
        cfg.train.select_on,
    )
    if not ctx.is_main:
        return best
    score = getattr(video, cfg.train.select_on)
    directory = cfg.checkpoint_path
    if score is not None and score > best:
        best = float(score)
        save_checkpoint(directory / "best.pt", network, objective, optimisation, epoch, best, cfg)
    save_checkpoint(directory / "last.pt", network, objective, optimisation, epoch, best, cfg)
    return best


def _log_counts(datasets: dict[str, Any], logger: RunLogger) -> None:
    """Clip and video counts per split."""
    parts = getattr(datasets["train"], "datasets", [datasets["train"]])
    logger.info(
        "clips (videos): train %s | val %d (%d) | test %s",
        " ".join(f"{d.name}={len(d)} ({len(d.video_ids)})" for d in parts),
        len(datasets["val"]),
        len(datasets["val"].video_ids),
        " ".join(f"{name}={len(d)} ({len(d.video_ids)})" for name, d in datasets["test"].items()),
    )


def train(detector: BaseDetector, ctx: DistributedContext) -> dict[str, Any]:  # noqa: PLR0915
    """Train detector, select on validation, then score the test battery."""
    cfg = detector.cfg
    seed_everything(cfg.train.seed, ctx.rank)
    directory = cfg.checkpoint_path
    if (directory / "best.pt").exists():
        raise FileExistsError(
            f"{directory / 'best.pt'} exists; pass --experiment <new-name> so this run does not overwrite it"
        )
    logger = RunLogger(
        cfg.log, to_dict(cfg), is_main=ctx.is_main, run_dir=directory, model_name=cfg.model.name, group=cfg.log.group
    )

    loaders, datasets = build_loaders(cfg, ctx)
    _log_counts(datasets, logger)
    network = detector.build_network().to(ctx.device)
    # The classifier is drawn from its own stream, so changing any earlier module
    # does not also change where the head starts.
    head_size = reseed_linear(cast(nn.Linear, network.classifier), cfg.train.seed)
    parameters = [p for p in network.parameters() if p.requires_grad]
    logger.info(
        "trainable parameters: %.2fM (classifier re-drawn from seed %d, %d parameters)",
        sum(p.numel() for p in parameters) / 1e6,
        cfg.train.seed,
        head_size,
    )

    weights = class_weights_for(datasets["train"], cfg)
    objective = detector.build_objective(weights).to(ctx.device)
    model: nn.Module = network
    if ctx.enabled:
        model = DistributedDataParallel(
            network,
            device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
            find_unused_parameters=detector.dynamic_graph,
        )

    optimizer, scheduler = _optimisation(cfg, parameters)
    amp_dtype = AMP_DTYPES[cfg.train.amp_dtype]
    use_amp = ctx.device.type == "cuda"
    accumulation = max(1, cfg.train.batch_size // (cfg.train.micro_batch_size * ctx.world_size))
    logger.info("gradient accumulation: %d micro-batches -> effective batch %d", accumulation, cfg.train.batch_size)

    footprint = Footprint(detector.base_model, detector.relation, ctx)
    footprint.start("train")
    step, best, stop = 0, -1.0, False
    for epoch in range(cfg.train.epochs):
        epoch_started = time.perf_counter()
        model.train()
        if ctx.enabled:
            loaders["train"].sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)

        for micro, batch in enumerate(loaders["train"]):
            frames = batch.frames.to(ctx.device, non_blocking=True)
            labels = batch.label.to(ctx.device, non_blocking=True)
            # valid marks padded frames (short videos repeat their last frame up to
            # n_frames); each network decides whether to use it.
            valid = frame_validity(batch.n_valid, frames.shape[1], ctx.device)
            with torch.autocast(ctx.device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(frames, valid=valid)
            loss, parts = objective(output, labels)
            (loss / accumulation).backward()
            if (micro + 1) % accumulation:
                continue

            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, cfg.train.grad_clip or float("inf")))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            sync_buffers_(objective, ctx)
            step += 1
            if step % cfg.log.log_every_steps == 0:
                scalars = {**parts, "grad_norm": grad_norm, **detector.train_stats(network)}
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    "epoch %d step %d | %s | lr %.2e",
                    epoch,
                    step,
                    " ".join(f"{name} {value:.4f}" for name, value in scalars.items()),
                    lr,
                )
                logger.log_metrics(
                    {**{f"train/{name}": value for name, value in scalars.items()}, "train/lr": lr, "epoch": epoch},
                    step=step,
                    table="train",
                )
            stop = bool(cfg.train.max_steps and step >= cfg.train.max_steps)
            if stop:
                break

        if (epoch + 1) % cfg.train.eval_every_epochs == 0 or stop or epoch == cfg.train.epochs - 1:
            optimizer.zero_grad(set_to_none=True)
            logger.info("epoch %d trained in %.1f min", epoch, (time.perf_counter() - epoch_started) / 60)
            best = _select_and_save(
                model,
                network,
                objective,
                loaders["val"],
                ctx,
                cfg,
                logger,
                (optimizer, scheduler),
                epoch,
                step,
                best,
            )
        if stop:
            break
    footprint.stop("train")

    # Rank 0 wrote best.pt; the barrier makes it visible to the other ranks.
    ctx.barrier()
    best_checkpoint = directory / "best.pt"
    if best_checkpoint.exists():
        load_trainable_weights(network, str(best_checkpoint), ctx.device)
        logger.info("loaded best checkpoint (val %s=%.4f)", cfg.train.select_on, best)

    results, protocols, source = score_battery(model, loaders["val"], loaders["test"], ctx, cfg, footprint)
    row: dict[str, float] = {"test/source_threshold": source.value}
    for name, blocks in protocols.items():
        row |= flat(f"test/{name}", blocks)
    logger.log_metrics(row | footprint.metrics(), step=step, table="test")
    footprint.write(cfg.results_path)
    logger.finish()
    return results
