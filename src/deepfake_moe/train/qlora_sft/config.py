"""YAML/CLI-backed config for the two-stage QLoRA SFT pipeline.

Stage selection is entirely config/CLI driven (``tuning_mode``); there is no
separate package per stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from deepfake_moe.utils.env import expand_env


_TUNING_MODES = ("thinker_lora", "modality_projectors")

# HF TrainingArguments fields that must stay string-typed end-to-end: YAML/CLI
# parsing must preserve "no" as the *string* "no" (e.g. for eval_strategy /
# save_strategy), not coerce it to boolean False.
_STRING_ONLY_FIELDS = frozenset({"eval_strategy", "save_strategy"})


@dataclass
class QLoraSFTConfig:
    """Typed config for the two-stage QLoRA SFT YAML files."""

    # ── Stage selection ──────────────────────────────────────────────────────
    tuning_mode: str  # "thinker_lora" (Stage 1) | "modality_projectors" (Stage 2)
    init_checkpoint: str | None = None  # prior stage's final_artifact; starts a NEW stage
    resume_from_checkpoint: str | None = None  # resumes an INTERRUPTED run of the CURRENT stage

    # ── Base model ───────────────────────────────────────────────────────────
    model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    attn_implementation: str = "sdpa"  # qlora_train.yaml overrides to flash_attention_2 for Stage 1

    # ── Data (paths from .env) ──────────────────────────────────────────────
    manifest_dir: str = ""
    data_root: str = ""
    dataset_video_roots: dict[str, str] = field(default_factory=dict)
    train_dataset_sources: dict[str, list[str]] = field(default_factory=dict)
    val_dataset_sources: dict[str, list[str]] = field(default_factory=dict)
    train_samples_per_dataset: int = 5000
    val_samples_per_dataset: int = 2000
    audio_sample_rate: int = 16000
    num_frames: int = 64
    max_seq_len: int = 4096
    seed: int = 42

    # ── LoRA (Stage 1 only; ignored for modality_projectors) ───────────────────
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # ── Optimization ─────────────────────────────────────────────────────────
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_steps: int = -1
    num_train_epochs: float = 1.0
    max_grad_norm: float = 1.0

    # ── Loop mechanics (mapped onto HF TrainingArguments in train.py) ────────
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 10
    save_total_limit: int = 3
    dataloader_num_workers: int = 2
    # Checkpoint selection: best (lowest) validation CE, owned by
    # SFTTrainer.evaluate — never Trainer's load_best_model_at_end reload
    # (crashes on peft/transformers version skew, see train.py's docstring).
    load_best_model_at_end: bool = True

    # ── Output / merge ───────────────────────────────────────────────────────
    output_dir: str = "checkpoints"
    merge_after_train: bool = False

    # ── Reproducibility / run naming ────────────────────────────────────────
    run_name: str = ""
    wandb_project: str = "deepfake-moe"
    wandb_group: str = ""
    wandb_tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate stage-dependent invariants."""
        if self.tuning_mode not in _TUNING_MODES:
            raise ValueError(f"tuning_mode must be one of {_TUNING_MODES}, got {self.tuning_mode!r}")
        if self.tuning_mode == "modality_projectors" and not self.init_checkpoint:
            raise ValueError(
                "tuning_mode='modality_projectors' (Stage 2) requires init_checkpoint pointing "
                "at Stage 1's final_artifact — Stage 2 freezes Stage 1's LoRA, it doesn't start fresh."
            )

    @property
    def effective_batch_size(self) -> int:
        """Micro-batch × gradient accumulation (model-parallel, not data-parallel)."""
        return self.micro_batch_size * self.gradient_accumulation_steps

    @classmethod
    def from_dict(cls, data: dict) -> "QLoraSFTConfig":
        """Build from a mapping, expanding ``${VAR}`` and rejecting unknown keys."""
        data = expand_env(dict(data))
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown QLoraSFTConfig keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, object] | None = None) -> "QLoraSFTConfig":
        """Load from YAML, then apply ``--set key=value`` overrides (already parsed)."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        if overrides:
            data = {**data, **overrides}
        return cls.from_dict(data)


def parse_set_override(key: str, raw_value: str) -> object:
    """Coerce one ``--set key=value`` CLI override to a typed value.

    ``key`` is checked against :data:`_STRING_ONLY_FIELDS` first so
    ``eval_strategy=no`` / ``save_strategy=no`` stay the *string* ``"no"``
    (HF's ``TrainingArguments`` treats these as enum-like strings, not
    booleans) — re-coercing them to ``False`` would be a real bug.

    Otherwise the value is parsed with ``yaml.safe_load`` for int/float/bool/
    null/list coercion, falling back to the raw string if that fails or
    yields a non-scalar.
    """
    if key in _STRING_ONLY_FIELDS:
        return raw_value
    try:
        parsed = yaml.safe_load(raw_value)
    except yaml.YAMLError:
        return raw_value
    if isinstance(parsed, (dict, list)):
        return raw_value
    return parsed


def parse_set_overrides(pairs: list[str]) -> dict[str, object]:
    """Parse a list of ``key=value`` CLI arguments into a typed override dict."""
    overrides: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got {pair!r}")
        key, raw_value = pair.split("=", 1)
        overrides[key] = parse_set_override(key, raw_value)
    return overrides
