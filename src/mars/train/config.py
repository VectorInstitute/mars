"""Run configuration shared by every model: data, recipe, selection and logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mars.data.config import DataConfig


SELECTION = ("roc_auc", "pr_auc", "accuracy")


@dataclass
class ModelConfig:
    """Fields every model config has; a model subclasses this with its own."""

    name: str = ""
    n_classes: int = 2


@dataclass
class TrainConfig:
    """Optimizer, schedule, seed and checkpoint selection."""

    epochs: int = 3
    batch_size: int = 32
    micro_batch_size: int = 8
    eval_batch_size: int | None = None
    lr: float = 1e-4
    weight_decay: float = 1.0e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    grad_clip: float | None = 1.0
    decay_steps: int = 1000
    decay_rate: float = 0.5
    class_balance: str = "none"
    amp_dtype: str = "bfloat16"
    seed: int = 0
    max_steps: int | None = None
    eval_every_epochs: int = 1
    select_on: str = "roc_auc"


@dataclass
class LogConfig:
    """Checkpoints, result folders, and W&B."""

    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    group: str = ""
    run_name: str = ""
    wandb: bool = True
    wandb_project: str = "mars"
    wandb_entity: str | None = None
    log_every_steps: int = 25


@dataclass
class RunConfig:
    """One run: shared data, recipe and logging, plus the model's own section."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    log: LogConfig = field(default_factory=LogConfig)

    @property
    def checkpoint_path(self) -> Path:
        """Directory for best.pt and last.pt."""
        return self._run_path(self.log.checkpoint_dir)

    @property
    def results_path(self) -> Path:
        """Directory for this run's per-corpus results."""
        return self._run_path(self.log.results_dir)

    def _run_path(self, root: str) -> Path:
        """{root}/{model}/{group}/{run_name}, with group optional."""
        parts = [root, self.model.name, self.log.group, self.log.run_name]
        return Path(*[p for p in parts if p])


def validate_run(cfg: RunConfig) -> None:
    """Raise ValueError if the shared sections are inconsistent."""
    d, t = cfg.data, cfg.train
    if not d.train:
        raise ValueError("data.train is required")
    if not d.test:
        raise ValueError("data.test is required (e.g. test: [ffpp])")
    if not cfg.log.run_name:
        raise ValueError("log.run_name is required (set it in the model YAML or pass --experiment)")
    if d.padding not in ("last_frame", "zeros"):
        raise ValueError(f"unknown padding {d.padding!r}")
    if t.class_balance not in ("none", "weighted_ce", "sampler"):
        raise ValueError(f"unknown class_balance {t.class_balance!r}")
    if t.batch_size % t.micro_batch_size:
        raise ValueError(f"batch_size {t.batch_size} not divisible by micro_batch_size {t.micro_batch_size}")
    if t.select_on not in SELECTION:
        raise ValueError(f"train.select_on must be one of {SELECTION}, got {t.select_on!r}")
