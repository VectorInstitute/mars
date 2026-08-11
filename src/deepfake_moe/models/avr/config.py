"""YAML/CLI-backed config for the AVR module, mirroring ``QLoraSFTConfig``'s shape."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from deepfake_moe.utils.env import expand_env


@dataclass
class AVRConfig:
    """Typed config for the AVR module (encoders -> masking -> ... -> fusion)."""

    # ── Base model / frozen encoders ────────────────────────────────────────
    model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    # ── Architecture ─────────────────────────────────────────────────────────
    hidden_dim: int = 2048  # Thinker hidden size; also the frozen encoders' output dim
    num_experts: int = 4
    top_k: int = 2
    expert_ffn_dim: int = 4096
    imputation_num_heads: int = 8
    imputation_timestamp_bias: bool = True
    # Discretized time-bucket lookup table for the masked-reconstruction
    # query (see imputation.py's MaskedQueryEmbedding) — not sinusoidal,
    # since clip durations here are bounded and known in advance.
    mask_query_bucket_seconds: float = 0.5
    mask_query_max_seconds: float = 30.0

    # ── Masking (Stage 1 only, see masking.py) ──────────────────────────────
    mask_window_seconds: float = 1.0

    # ── Stage 1 loss ─────────────────────────────────────────────────────────
    gaussian_nll_warmup_steps: int = 0
    # Reserved: the contrastive term is not part of Stage 1's first pass,
    # reconstruction NLL alone is.
    contrastive_temperature: float = 0.07
    router_aux_loss_coef: float = 0.01

    # ── Data (paths from .env) ────────────────────────────────────────────────
    manifest_dir: str = ""
    data_root: str = ""
    dataset_video_roots: dict[str, str] = field(default_factory=dict)
    train_dataset_sources: dict[str, list[str]] = field(default_factory=dict)
    val_dataset_sources: dict[str, list[str]] = field(default_factory=dict)
    # No default: real-only pool sizes are very uneven across datasets, and
    # a silent default would let one dataset dominate without a deliberate
    # choice.
    train_samples_per_dataset: int = field(kw_only=True)
    val_samples_per_dataset: int = field(kw_only=True)
    audio_sample_rate: int = 16000
    num_frames: int = 64
    seed: int = 42

    # ── Optimization ─────────────────────────────────────────────────────────
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_steps: int = -1
    num_train_epochs: float = 1.0
    max_grad_norm: float = 1.0

    # ── Loop mechanics ───────────────────────────────────────────────────────
    # Best checkpoint (lowest validation loss) is saved on this cadence; no
    # separate periodic/resume checkpoint exists yet.
    eval_steps: int = 200
    logging_steps: int = 10
    dataloader_num_workers: int = 2

    # ── Output / run naming ──────────────────────────────────────────────────
    output_dir: str = "checkpoints"
    run_name: str = ""
    wandb_project: str = ""
    wandb_group: str = ""
    wandb_tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate architecture invariants shared by every consumer of this config."""
        if self.num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {self.num_experts}")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError(f"top_k must be in [1, num_experts={self.num_experts}], got {self.top_k}")
        if self.hidden_dim % self.imputation_num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"imputation_num_heads ({self.imputation_num_heads})"
            )
        if self.mask_window_seconds <= 0:
            raise ValueError(f"mask_window_seconds must be > 0, got {self.mask_window_seconds}")
        if self.mask_query_bucket_seconds <= 0:
            raise ValueError(f"mask_query_bucket_seconds must be > 0, got {self.mask_query_bucket_seconds}")
        if self.mask_query_max_seconds <= 0:
            raise ValueError(f"mask_query_max_seconds must be > 0, got {self.mask_query_max_seconds}")
        if self.train_samples_per_dataset <= 0:
            raise ValueError(f"train_samples_per_dataset must be > 0, got {self.train_samples_per_dataset}")
        if self.val_samples_per_dataset <= 0:
            raise ValueError(f"val_samples_per_dataset must be > 0, got {self.val_samples_per_dataset}")
        if self.eval_steps <= 0:
            raise ValueError(f"eval_steps must be > 0, got {self.eval_steps}")
        if self.logging_steps <= 0:
            raise ValueError(f"logging_steps must be > 0, got {self.logging_steps}")

    @classmethod
    def from_dict(cls, data: dict) -> AVRConfig:
        """Build from a mapping, expanding ``${VAR}`` and rejecting unknown keys."""
        data = expand_env(dict(data))
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown AVRConfig keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, object] | None = None) -> AVRConfig:
        """Load from YAML, then apply already-parsed ``--set key=value`` overrides."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        if overrides:
            data = {**data, **overrides}
        return cls.from_dict(data)
