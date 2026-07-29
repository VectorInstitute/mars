"""CLI entry point for AVR Stage 1 self-supervised pretraining.

Kept separate from ``run_train.py``'s ``tuning_mode`` dispatch: Stage 1 uses
its own config class, trainer, and data shape, and never touches the
Thinker decoder. The training mechanics (``AVRSSLTrainer``) live in
``train/avr_ssl/trainer.py``; this module only wires the config together
and drives the run.

    uv run python -m deepfake_moe.run_train_avr_ssl \\
        --config configs/avr_ssl.yaml [--set key=value ...]
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

from deepfake_moe.models.avr.config import AVRConfig
from deepfake_moe.train.avr_ssl.trainer import AVRSSLTrainer
from deepfake_moe.train.qlora_sft.config import parse_set_overrides
from deepfake_moe.utils.logging_utils import get_logger


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="AVR Stage 1: self-supervised pretraining")
    p.add_argument("--config", required=True, help="Path to configs/avr_ssl.yaml")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="key=value",
        help="Override a config field (repeatable).",
    )
    return p.parse_args()


def main() -> None:
    """Load config, build the trainer, and run Stage 1 training."""
    args = parse_args()
    overrides = parse_set_overrides(args.overrides)
    cfg = AVRConfig.from_yaml(args.config, overrides)
    logger.info(
        "AVR Stage 1  train_samples/ds=%d  val_samples/ds=%d  num_experts=%d  top_k=%d",
        cfg.train_samples_per_dataset,
        cfg.val_samples_per_dataset,
        cfg.num_experts,
        cfg.top_k,
    )

    if cfg.wandb_project:
        import wandb  # noqa: PLC0415

        wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group or None,
            name=cfg.run_name or None,
            tags=cfg.wandb_tags or None,
            config=asdict(cfg),
        )

    trainer = AVRSSLTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
