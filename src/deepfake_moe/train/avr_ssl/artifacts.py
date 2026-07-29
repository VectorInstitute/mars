"""Stage 1 checkpoint save/load: a plain state dict, no PEFT machinery.

Only imputation, router, and expert weights are saved — the aggregate/
project head is not part of Stage 1 (see ``trainer.py``'s
``freeze_aggregate_project``), so its random-init weights are excluded
rather than checkpointed.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from deepfake_moe.models.avr.config import AVRConfig
from deepfake_moe.models.avr.model import AVRModel
from deepfake_moe.utils.logging_utils import get_logger


logger = get_logger(__name__)

TRAINABLE_MODULE_PREFIXES = ("imputation.", "router.", "experts.")


def _trainable_state_dict(model: AVRModel) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if k.startswith(TRAINABLE_MODULE_PREFIXES)}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def save_artifact(
    model: AVRModel,
    cfg: AVRConfig,
    artifact_dir: str | Path,
    *,
    reproducibility: dict[str, Any],
) -> None:
    """Save imputation, router, and expert weights plus run metadata.

    Examples
    --------
    >>> import tempfile
    >>> from deepfake_moe.models.avr.config import AVRConfig
    >>> from deepfake_moe.models.avr.model import AVRModel
    >>> cfg = AVRConfig(hidden_dim=4, imputation_num_heads=2, expert_ffn_dim=8,
    ...                  train_samples_per_dataset=1, val_samples_per_dataset=1)
    >>> model = AVRModel(cfg)
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     save_artifact(model, cfg, tmp, reproducibility={"global_step": 1})
    ...     sorted(p.name for p in Path(tmp).iterdir())
    ['avr_model.pt', 'meta.json']
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_trainable_state_dict(model), artifact_dir / "avr_model.pt")
    meta = {"config": asdict(cfg), "git_commit": _git_commit(), **reproducibility}
    (artifact_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Saved AVR Stage 1 artifact to %s", artifact_dir)


def load_avr_ssl_weights(model: AVRModel, artifact_dir: str | Path) -> None:
    """Load imputation, router, and expert weights from a Stage 1 artifact."""
    state_dict = torch.load(Path(artifact_dir) / "avr_model.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    real_missing = [k for k in missing if k.startswith(TRAINABLE_MODULE_PREFIXES)]
    if real_missing:
        raise RuntimeError(f"Missing expected AVR Stage 1 weights: {real_missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys in AVR Stage 1 artifact: {unexpected}")
    logger.info("Loaded AVR Stage 1 weights from %s", artifact_dir)
