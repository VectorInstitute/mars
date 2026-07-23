"""CLI entry point for two-stage QLoRA SFT training.

Stage is 100% config/CLI driven via ``tuning_mode`` — this module does not
know or care which stage it's running; ``qlora_sft/model.py`` dispatches on
it. The actual training mechanics (``SFTTrainer``: loss, checkpoint
selection) live in ``qlora_sft/trainer.py``; this module only wires config,
data, and model together and drives the run, mirroring how ``run_eval.py``
calls into ``eval/runner.py``.

    uv run python -m deepfake_moe.run_train \\
        --config configs/qlora_train.yaml [--set key=value ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from deepfake_moe.train.common.collator import SFTCollator
from deepfake_moe.train.common.data import SFTDataset
from deepfake_moe.train.common.splits import load_split_entries
from deepfake_moe.train.qlora_sft.artifacts import save_artifact
from deepfake_moe.train.qlora_sft.config import QLoraSFTConfig, parse_set_overrides
from deepfake_moe.train.qlora_sft.model import build_model, load_processor
from deepfake_moe.train.qlora_sft.trainer import _BEST_ARTIFACT_SUBDIR, SFTTrainer, build_training_arguments
from deepfake_moe.utils.logging_utils import get_logger
from deepfake_moe.utils.paths import default_audio_dir


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Two-stage QLoRA SFT (thinker_lora | modality_projectors)")
    p.add_argument("--config", required=True, help="Path to configs/qlora_train.yaml or configs/projector_train.yaml")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="key=value",
        help="Override a config field (repeatable). 'no'/'yes' stay strings — see config.py.",
    )
    return p.parse_args()


def _group_by_dataset(entries: list[dict]) -> dict[str, list[dict]]:
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_dataset[entry.get("dataset", "")].append(entry)
    return dict(by_dataset)


def _build_dataset(entries: list[dict], cfg: QLoraSFTConfig, audio_dir: str) -> SFTDataset:
    """Build an ``SFTDataset`` from entries with the shared clip-decode settings."""
    return SFTDataset(
        entries,
        num_frames=cfg.num_frames,
        audio_sample_rate=cfg.audio_sample_rate,
        data_root=cfg.data_root,
        audio_dir=audio_dir,
        dataset_video_roots=cfg.dataset_video_roots,
    )


def _finalize_artifact(trainer: SFTTrainer, cfg: QLoraSFTConfig, run_dir: Path, reproducibility: dict) -> Path:
    """Write ``final_artifact/`` from the best snapshot, else the last weights."""
    final_dir = run_dir / "final_artifact"
    best_dir = run_dir / _BEST_ARTIFACT_SUBDIR

    if cfg.load_best_model_at_end and best_dir.exists():
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.copytree(best_dir, final_dir)
        # Refresh meta with the full reproducibility block (best snapshot's was lean).
        meta = {
            "tuning_mode": cfg.tuning_mode,
            "model_id": cfg.model_id,
            "init_checkpoint": cfg.init_checkpoint,
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "lora_target_modules": cfg.lora_target_modules,
            "reproducibility": {**reproducibility, "best_step": trainer._best_step},
        }
        (final_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        logger.info("Finalized from best snapshot (step=%s) -> %s", trainer._best_step, final_dir)
        return final_dir

    save_artifact(trainer.model, cfg, final_dir, reproducibility=reproducibility)
    logger.info("Finalized from last weights -> %s", final_dir)
    return final_dir


def main() -> None:
    """CLI entry point: load config, build model + data, train, save the artifact."""
    args = parse_args()
    overrides = parse_set_overrides(args.overrides)
    cfg = QLoraSFTConfig.from_yaml(args.config, overrides)
    logger.info(
        "tuning_mode=%s  effective_batch_size=%d  train_samples/ds=%d  val_samples/ds=%d",
        cfg.tuning_mode,
        cfg.effective_batch_size,
        cfg.train_samples_per_dataset,
        cfg.val_samples_per_dataset,
    )

    processor = load_processor(cfg)
    model = build_model(cfg)

    train_entries = load_split_entries(
        cfg.manifest_dir, cfg.train_dataset_sources, cfg.train_samples_per_dataset, seed=cfg.seed, stratify=True
    )
    val_entries = load_split_entries(
        cfg.manifest_dir, cfg.val_dataset_sources, cfg.val_samples_per_dataset, seed=cfg.seed, stratify=False
    )
    val_by_dataset = _group_by_dataset(val_entries)

    audio_dir = default_audio_dir(cfg.data_root)
    train_dataset = _build_dataset(train_entries, cfg, audio_dir)
    val_dataset = _build_dataset(val_entries, cfg, audio_dir)

    collator = SFTCollator(processor, use_audio_in_video=False)

    run_dir = Path(cfg.output_dir) / (cfg.run_name or cfg.tuning_mode)

    if cfg.wandb_project:
        import wandb  # noqa: PLC0415

        wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group or None,
            name=cfg.run_name or None,
            tags=cfg.wandb_tags or None,
            config=asdict(cfg),
        )

    trainer = SFTTrainer(
        model=model,
        args=build_training_arguments(cfg, run_dir),
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        sft_cfg=cfg,
        val_by_dataset=val_by_dataset,
        sft_collator=collator,
    )

    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint or None)

    reproducibility = {
        "seed": cfg.seed,
        "effective_batch_size": cfg.effective_batch_size,
        "total_optimizer_steps": trainer.state.global_step,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "warmup_ratio": cfg.warmup_ratio,
        "max_seq_len": cfg.max_seq_len,
        "num_frames": cfg.num_frames,
        "best_metric": trainer._best_metric,
    }

    final_dir = _finalize_artifact(trainer, cfg, run_dir, reproducibility)
    logger.info("Training complete. Final artifact -> %s", final_dir)

    if cfg.merge_after_train:
        from deepfake_moe.train.qlora_sft.merge import merge_artifact  # noqa: PLC0415

        merge_artifact(cfg, final_dir)


if __name__ == "__main__":
    main()
