"""Console, CSV, and W&B logging."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any


def get_logger(name: str = "mars", rank: int = 0) -> logging.Logger:
    """Return a process-local logger. Non-zero ranks log errors only."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    logger.propagate = False
    return logger


class RunLogger:
    """Write metrics to CSV and, when enabled, to Weights & Biases."""

    def __init__(
        self,
        cfg: Any,
        config_dict: dict[str, Any],
        is_main: bool = True,
        run_dir: Path | None = None,
        model_name: str = "",
        group: str = "",
    ) -> None:
        self.is_main = is_main
        if run_dir is not None:
            self.run_dir = Path(run_dir)
        else:
            self.run_dir = Path(cfg.checkpoint_dir) / model_name / cfg.run_name
        self.log = get_logger(rank=0 if is_main else 1)
        self._wandb: Any = None
        self._csv_fields: dict[Path, list[str]] = {}
        if not is_main:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if cfg.wandb:
            try:
                import wandb

                self._wandb = wandb
                wandb.init(
                    project=cfg.wandb_project,
                    entity=cfg.wandb_entity,
                    name=cfg.run_name,
                    group=group or None,
                    dir=str(self.run_dir),
                    config=config_dict,
                )
            except Exception as exc:
                self.log.warning("W&B disabled (%s); CSV logging continues", exc)
                self._wandb = None

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        table: str = "metrics",
    ) -> None:
        """Append ``metrics`` to W&B and to ``{table}.csv``."""
        if not self.is_main:
            return
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
        self._append_csv(self.run_dir / f"{table}.csv", {"step": step, **metrics})

    def _append_csv(self, path: Path, row: dict[str, Any]) -> None:
        fields = self._csv_fields.get(path)
        if fields is None:
            fields = list(row)
            self._csv_fields[path] = fields
            with path.open("w", newline="") as handle:
                csv.DictWriter(handle, fieldnames=fields).writeheader()
        with path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore").writerow(row)

    def info(self, message: str, *args: Any) -> None:
        """Log an info line on the main rank."""
        if self.is_main:
            self.log.info(message, *args)

    def warning(self, message: str, *args: Any) -> None:
        """Log a warning on the main rank."""
        if self.is_main:
            self.log.warning(message, *args)

    def finish(self) -> None:
        """Close the W&B run if one was started."""
        if self.is_main and self._wandb is not None:
            self._wandb.finish()
