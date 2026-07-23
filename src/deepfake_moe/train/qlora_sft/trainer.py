"""``SFTTrainer``: the sharded Qwen3-Omni thinker's HF ``Trainer`` subclass.

Because the directly-loaded, ``device_map="auto"`` Thinker (see ``model.py``)
is the exact object we train and carries ``hf_device_map`` itself, HF
``Trainer`` recognizes it as already multi-GPU model-parallel and drives the
loop, checkpointing, and grad-norm logging for us.

Loss (train and val) is computed via an lm_head pre-hook that projects only the
answer positions and takes cross-entropy directly — ``labels=`` is never passed
to the model. This avoids the full-sequence 150k-vocab logit tensor and the
model's internal cross-device ``nll_loss`` that stalled/asserted on the sharded
thinker. Generation-based macro-F1 / AUC is scored separately by the vLLM eval
harness on saved checkpoints, not in the training loop.

Checkpoint selection is handled HERE: best = lowest validation CE, snapshotted
via ``artifacts.save_artifact``. Trainer's ``load_best_model_at_end`` reload is
NOT used — it goes through ``set_peft_model_state_dict`` →
``convert_peft_adapter_state_dict_for_transformers``, which crashes on
peft/transformers version skew. Saving our own snapshot on the way UP never
touches that load path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import Trainer, TrainingArguments

from deepfake_moe.models.hf import device_routing
from deepfake_moe.train.common.collator import SFTCollator
from deepfake_moe.train.qlora_sft.artifacts import save_artifact
from deepfake_moe.train.qlora_sft.config import QLoraSFTConfig
from deepfake_moe.utils.logging_utils import get_logger


logger = get_logger(__name__)

_BEST_ARTIFACT_SUBDIR = "best_artifact"


def build_training_arguments(cfg: QLoraSFTConfig, run_dir: Path) -> TrainingArguments:
    """Map ``QLoraSFTConfig`` onto ``TrainingArguments``.

    ``gradient_checkpointing`` is left False here because
    ``prepare_model_for_kbit_training(..., use_gradient_checkpointing=True)``
    already enabled it (and installed the input-require-grads hook that keeps
    the QLoRA path differentiable). ``prediction_loss_only`` keeps the
    teacher-forced eval pass from gathering the 152k-vocab logits.
    ``remove_unused_columns=False`` stops Trainer from stripping the custom
    dataset keys (``clip``/``target_text``/``user_prompt``) the collator needs.
    ``load_best_model_at_end`` / ``metric_for_best_model`` are intentionally
    NOT set — best selection is owned by ``SFTTrainer.evaluate`` to avoid
    Trainer's version-fragile PEFT reload path.
    """
    return TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=cfg.micro_batch_size,
        per_device_eval_batch_size=cfg.micro_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        max_steps=cfg.max_steps if cfg.max_steps and cfg.max_steps > 0 else -1,
        num_train_epochs=cfg.num_train_epochs,
        max_grad_norm=cfg.max_grad_norm,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=False,
        bf16=True,
        gradient_checkpointing=False,
        prediction_loss_only=True,
        dataloader_num_workers=cfg.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=(["wandb"] if cfg.wandb_project else []),
        run_name=cfg.run_name or None,
        seed=cfg.seed,
    )


class SFTTrainer(Trainer):
    """``Trainer`` for the sharded Qwen3-Omni thinker, with a stability-safe loss.

    - ``_prepare_inputs`` delegates to :mod:`device_routing` so every batch
      lands on the sharded thinker's anchor device.
    - ``compute_loss`` / ``evaluate`` both go through ``_masked_lm_loss``, which
      projects ``lm_head`` on ONLY the answer positions and computes cross-entropy
      itself. This avoids the full-sequence 150k-vocab logit tensor and the
      model's internal cross-device ``nll_loss`` — the two things that stalled and
      asserted on the sharded model. ``labels=`` is never passed to the model.
    - No in-loop generation: macro-F1 / AUC are scored separately by the vLLM
      eval harness on saved checkpoints. Best checkpoint = lowest validation CE,
      snapshotted via ``save_artifact`` (never the version-fragile PEFT reload).
    """

    def __init__(
        self,
        *args: Any,
        sft_cfg: QLoraSFTConfig,
        val_by_dataset: dict[str, list[dict]],
        sft_collator: SFTCollator,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._sft_cfg = sft_cfg
        self._val_by_dataset = val_by_dataset
        self._sft_collator = sft_collator
        self._best_metric: float | None = None
        self._best_step: int | None = None
        if self.args.output_dir is None:
            raise ValueError("TrainingArguments.output_dir must be set")
        self._best_dir = Path(self.args.output_dir) / _BEST_ARTIFACT_SUBDIR

    def _model(self) -> torch.nn.Module:
        """Return ``self.model``, narrowed to non-``None`` for the type checker."""
        if self.model is None:
            raise RuntimeError("Trainer.model is not set")
        return self.model

    def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return device_routing.prepare_inputs(self.model, dict(inputs))

    def _lm_head(self) -> Any:
        """Resolve (and cache) the thinker's ``lm_head`` through the PEFT wrapper."""
        if getattr(self, "_lm_head_module", None) is None:
            for name, module in self._model().named_modules():
                if name.endswith("lm_head"):
                    self._lm_head_module = module
                    break
            else:
                raise RuntimeError("lm_head not found on model — cannot install pre-hook")
        return self._lm_head_module

    def _masked_lm_loss(self, model: Any, inputs: dict[str, Any]) -> torch.Tensor:
        """Cross-entropy over ONLY the supervised tokens, via an lm_head pre-hook.

        Passing ``labels=`` to the Qwen3-Omni thinker runs ``lm_head`` over the
        full ~5k-token sequence (a ~150k-vocab fp32 logit tensor on one GPU) plus
        its internal cross-device ``nll_loss`` — the source of the stalls and the
        ``t < n_classes`` assert on the sharded model. Instead we slice the hidden
        states to the answer positions *before* ``lm_head`` runs and compute CE
        ourselves. Token at position ``i`` is predicted from the hidden state at
        ``i-1``. Assumes ``micro_batch_size == 1``.
        """
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        loss_positions = (labels[0] != -100).nonzero(as_tuple=False).squeeze(-1)
        hs_positions = (loss_positions - 1).clamp(min=0)

        captured: dict[str, Any] = {}

        def _slice(module: Any, args: tuple) -> tuple:
            hidden = args[0]
            # hidden may be [B, T, H] (3D) or [T, H] (2D); index the SEQUENCE dim.
            seq_dim = 1 if hidden.dim() == 3 else 0
            seq_len = hidden.shape[seq_dim]
            idx = hs_positions.to(hidden.device)
            if idx.numel() and int(idx.max()) >= seq_len:
                logger.warning("LENGTH MISMATCH hidden_seq=%d labels_len=%d — clamping", seq_len, int(labels.shape[1]))
                idx = idx.clamp(max=seq_len - 1)
            sliced = hidden.index_select(seq_dim, idx)  # keeps the grad graph to LoRA
            captured["hidden"] = sliced
            captured["module"] = module
            return (sliced,) + tuple(args[1:])

        handle = self._lm_head().register_forward_pre_hook(_slice)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(**inputs, use_cache=False)  # runs lm_head on the slice; its output is unused
        finally:
            handle.remove()

        # The model's lm_head runs under bf16 autocast and emits non-finite
        # logits even though the hidden states + weight are finite fp32 (DIAG
        # confirmed). Recompute the answer-position logits ourselves in fp32 —
        # numerically safe, and still backprops through `hidden` to the LoRA
        # params.
        hidden = captured["hidden"]
        hidden = (hidden[0] if hidden.dim() == 3 else hidden).float()  # [n_answer_tokens, H]
        lm = captured["module"]
        logits = torch.nn.functional.linear(
            hidden, lm.weight.float(), lm.bias.float() if getattr(lm, "bias", None) is not None else None
        )  # [n_answer_tokens, vocab]
        target = labels[0, loss_positions].to(logits.device)
        if not torch.isfinite(logits).all():
            logger.warning("non-finite logits AFTER fp32 recompute (n=%d) — deeper issue", int(loss_positions.numel()))
        # A target id outside [0, n_classes) triggers the nll `t < n_classes`
        # CUDA assert. Log the exact numbers (sync forces this to print before
        # the assert) and clamp so the run continues and we can see them.
        n_classes = logits.shape[-1]
        t_min = int(target.min()) if target.numel() else 0
        t_max = int(target.max()) if target.numel() else -1
        if t_min < 0 or t_max >= n_classes:
            logger.error(
                "TARGET OUT OF RANGE target=[%d,%d] n_classes=%d logits.shape=%s labels_len=%d n_pos=%d — clamping",
                t_min,
                t_max,
                n_classes,
                tuple(logits.shape),
                int(labels.shape[1]),
                int(loss_positions.numel()),
            )
            target = target.clamp(0, n_classes - 1)
        return torch.nn.functional.cross_entropy(logits, target)

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> Any:
        """Masked cross-entropy via ``_masked_lm_loss``, moved onto ``args.device``."""
        # Our CE runs on lm_head's device (e.g. cuda:3); Trainer requires the
        # loss on args.device (cuda:0). .to() stays differentiable, so backward
        # still flows back to the LoRA params on their own devices.
        loss = self._masked_lm_loss(model, inputs).to(self.args.device)
        return (loss, {}) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: Any | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Compute validation loss as the mean masked CE over the val split.

        Uses the same pre-hook path as training, so it never touches the
        internal-loss / generation paths that stalled and asserted. Generation-
        based macro-F1 / AUC is scored separately via the vLLM eval harness on
        saved checkpoints. Best checkpoint = lowest validation CE.
        """
        loader = self.get_eval_dataloader(eval_dataset)
        model = self._model()
        was_training = model.training
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for raw_inputs in loader:
                inputs = self._prepare_inputs(raw_inputs)
                total += float(self._masked_lm_loss(model, inputs).item())
                n += 1
        if was_training:
            model.train()

        val_loss = total / n if n else float("nan")
        metrics = {f"{metric_key_prefix}_loss": val_loss}
        self.log(metrics)
        logger.info("eval step=%d val_loss=%.4f (n=%d)", self.state.global_step, val_loss, n)

        # Best = lowest validation CE (a real number now). Snapshot via
        # save_artifact — never the version-fragile PEFT reload path.
        if n and (self._best_metric is None or val_loss < self._best_metric):
            self._best_metric = val_loss
            self._best_step = self.state.global_step
            save_artifact(
                model,
                self._sft_cfg,
                self._best_dir,
                reproducibility={"global_step": self.state.global_step, "best_val_loss": val_loss},
            )
            logger.info("New best val_loss=%.4f at step=%d -> %s", val_loss, self._best_step, self._best_dir)
        return metrics
