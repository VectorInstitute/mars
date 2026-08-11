"""AVR Stage 1: reconstruction-only Gaussian NLL training loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration, get_scheduler

from deepfake_moe.models.avr.config import AVRConfig
from deepfake_moe.models.avr.encoders import encode, load_frozen_encoders
from deepfake_moe.models.avr.masking import (
    audio_token_timestamps,
    complementary_window_mask,
    visual_token_timestamps,
)
from deepfake_moe.models.avr.model import AVRModel
from deepfake_moe.train.avr_ssl.artifacts import save_artifact
from deepfake_moe.train.avr_ssl.dataset import AVRSSLDataset, load_real_only_entries
from deepfake_moe.utils.logging_utils import get_logger
from deepfake_moe.utils.paths import default_audio_dir


logger = get_logger(__name__)

_BEST_ARTIFACT_SUBDIR = "best_artifact"
_FINAL_ARTIFACT_SUBDIR = "final_artifact"
_PROMPT = "<|im_start|>user\ndescribe<|im_end|>\n"


def freeze_aggregate_project(model: AVRModel) -> None:
    """Exclude aggregate_project from Stage 1; its weights stay at random init."""
    for param in model.aggregate_project.parameters():
        param.requires_grad = False


def gaussian_nll_loss(z_true: torch.Tensor, mu: torch.Tensor, log_sigma_sq: torch.Tensor) -> torch.Tensor:
    """Elementwise Gaussian NLL, averaged over masked tokens and hidden dims.

    ``log_sigma_sq`` is one scalar per token and broadcasts over ``mu``'s
    hidden dimension.

    Examples
    --------
    >>> import torch
    >>> z = torch.zeros(2, 3)
    >>> mu = torch.zeros(2, 3)
    >>> log_sigma_sq = torch.zeros(2, 1)
    >>> round(gaussian_nll_loss(z, mu, log_sigma_sq).item(), 4)
    0.0
    """
    sigma_sq = log_sigma_sq.exp()
    return (((z_true - mu) ** 2) / (2 * sigma_sq) + 0.5 * log_sigma_sq).mean()


def _collate_single(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pass one decoded clip through unchanged; AVR trains one clip per step."""
    if len(batch) != 1:
        raise ValueError(f"expected batch_size=1, got {len(batch)}")
    return batch[0]


@dataclass
class StepMetrics:
    """Per-step loss breakdown, for logging."""

    loss: float
    reconstruction_loss: float
    aux_loss: float
    n_masked_audio: int
    n_masked_visual: int


def compute_loss(
    item: dict[str, Any],
    *,
    processor: Qwen3OmniMoeProcessor,
    thinker: Qwen3OmniMoeThinkerForConditionalGeneration,
    avr: AVRModel,
    cfg: AVRConfig,
    global_step: int,
) -> tuple[torch.Tensor, StepMetrics]:
    """Run one clip through encode, masking, and AVRModel; return the Stage 1 loss.

    During warmup (``global_step < cfg.gaussian_nll_warmup_steps``), sigma
    is fixed to 1.0 (pure MSE), so no gradient reaches the sigma head until
    warmup ends.
    """
    audio_device = next(thinker.audio_tower.parameters()).device
    visual_device = next(thinker.visual.parameters()).device

    batch = processor(
        text=[_PROMPT],
        audio=[item["waveform"]],
        videos=[item["frames"]],
        fps=item["fps"],
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        outputs = encode(
            thinker,
            input_features=batch["input_features"].to(audio_device),
            feature_attention_mask=batch["feature_attention_mask"].to(audio_device),
            pixel_values_videos=batch["pixel_values_videos"].to(visual_device),
            video_grid_thw=batch["video_grid_thw"].to(visual_device),
        )

    device = audio_device
    avr_dtype = next(avr.parameters()).dtype
    z_audio = outputs.audio_hidden.to(device=device, dtype=avr_dtype)
    z_visual = outputs.video_hidden.to(device=device, dtype=avr_dtype)

    audio_ts = audio_token_timestamps(z_audio.shape[0], device=device)
    visual_ts = visual_token_timestamps(
        batch["video_grid_thw"].cpu(), batch["video_second_per_grid"], thinker.visual.spatial_merge_size
    ).to(device)
    audio_mask, visual_mask = complementary_window_mask(audio_ts, visual_ts, cfg.mask_window_seconds)

    result = avr(z_audio, z_visual, audio_ts, visual_ts, audio_mask, visual_mask)

    z_true = torch.cat([z_audio, z_visual], dim=0)
    combined_mask = torch.cat([audio_mask, visual_mask])
    z_true_masked = z_true[combined_mask].float()
    mu_masked = result.mu_star[combined_mask].float()
    log_sigma_sq_masked = result.log_sigma_star_sq[combined_mask].float()
    if global_step < cfg.gaussian_nll_warmup_steps:
        log_sigma_sq_masked = torch.zeros_like(log_sigma_sq_masked)

    reconstruction_loss = gaussian_nll_loss(z_true_masked, mu_masked, log_sigma_sq_masked)
    loss = reconstruction_loss + cfg.router_aux_loss_coef * result.aux_loss

    metrics = StepMetrics(
        loss=float(loss.item()),
        reconstruction_loss=float(reconstruction_loss.item()),
        aux_loss=float(result.aux_loss.item()),
        n_masked_audio=int(audio_mask.sum().item()),
        n_masked_visual=int(visual_mask.sum().item()),
    )
    return loss, metrics


class AVRSSLTrainer:
    """Owns Stage 1's encode, mask, AVRModel training loop and checkpointing."""

    def __init__(self, cfg: AVRConfig) -> None:
        self.cfg = cfg
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(cfg.model_id)
        self.thinker = load_frozen_encoders(cfg.model_id)
        self.device = next(self.thinker.audio_tower.parameters()).device

        self.avr = AVRModel(cfg).to(device=self.device, dtype=torch.bfloat16)
        freeze_aggregate_project(self.avr)

        audio_dir = default_audio_dir(cfg.data_root)
        train_entries = load_real_only_entries(
            cfg.manifest_dir, cfg.train_dataset_sources, cfg.train_samples_per_dataset, seed=cfg.seed
        )
        val_entries = load_real_only_entries(
            cfg.manifest_dir, cfg.val_dataset_sources, cfg.val_samples_per_dataset, seed=cfg.seed
        )
        self.train_dataset = AVRSSLDataset(
            train_entries,
            num_frames=cfg.num_frames,
            audio_sample_rate=cfg.audio_sample_rate,
            data_root=cfg.data_root,
            audio_dir=audio_dir,
            dataset_video_roots=cfg.dataset_video_roots,
        )
        self.val_dataset = AVRSSLDataset(
            val_entries,
            num_frames=cfg.num_frames,
            audio_sample_rate=cfg.audio_sample_rate,
            data_root=cfg.data_root,
            audio_dir=audio_dir,
            dataset_video_roots=cfg.dataset_video_roots,
        )

        trainable_params = [p for p in self.avr.parameters() if p.requires_grad]
        logger.info("AVR Stage 1 trainable parameters: %d", sum(p.numel() for p in trainable_params))
        self.optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)

        self.max_steps = self._resolve_max_steps()
        logger.info(
            "train_clips=%d  val_clips=%d  max_steps=%d", len(self.train_dataset), len(self.val_dataset), self.max_steps
        )
        self.scheduler = get_scheduler(
            cfg.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=int(cfg.warmup_ratio * self.max_steps),
            num_training_steps=self.max_steps,
        )

        run_dir = Path(cfg.output_dir) / cfg.run_name
        self.best_dir = run_dir / _BEST_ARTIFACT_SUBDIR
        self.final_dir = run_dir / _FINAL_ARTIFACT_SUBDIR
        self._best_val_loss: float | None = None

    def _resolve_max_steps(self) -> int:
        if self.cfg.max_steps and self.cfg.max_steps > 0:
            return self.cfg.max_steps
        steps_per_epoch = max(1, len(self.train_dataset) // self.cfg.gradient_accumulation_steps)
        return max(1, int(steps_per_epoch * self.cfg.num_train_epochs))

    def _loader(self, dataset: AVRSSLDataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=self.cfg.dataloader_num_workers,
            collate_fn=_collate_single,
        )

    def _log_wandb(self, metrics: dict[str, float], step: int) -> None:
        """Log to the active wandb run, if ``run_train_avr_ssl.py`` started one."""
        if wandb.run is not None:
            wandb.log(metrics, step=step)

    def evaluate(self, *, step: int | None = None) -> float:
        """Return the mean reconstruction loss over the val set (excludes aux loss)."""
        self.avr.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for item in self._loader(self.val_dataset, shuffle=False):
                _, metrics = compute_loss(
                    item,
                    processor=self.processor,
                    thinker=self.thinker,
                    avr=self.avr,
                    cfg=self.cfg,
                    global_step=self.max_steps,
                )
                total += metrics.reconstruction_loss
                n += 1
        self.avr.train()
        val_loss = total / n if n else float("nan")
        logger.info("eval  val_reconstruction_loss=%.4f (n=%d)", val_loss, n)
        self._log_wandb({"eval/reconstruction_loss": val_loss}, step if step is not None else self.max_steps)
        return val_loss

    def _maybe_checkpoint(self, step: int, val_loss: float) -> None:
        if self._best_val_loss is not None and val_loss >= self._best_val_loss:
            return
        self._best_val_loss = val_loss
        save_artifact(
            self.avr, self.cfg, self.best_dir, reproducibility={"global_step": step, "best_val_loss": val_loss}
        )
        logger.info("new best val_reconstruction_loss=%.4f at step=%d -> %s", val_loss, step, self.best_dir)

    def train(self) -> None:
        """Run the Stage 1 training loop, evaluating and checkpointing on a cadence."""
        self.avr.train()
        step, accum = 0, 0
        accum_loss = 0.0
        running_loss = 0.0
        self.optimizer.zero_grad()

        while step < self.max_steps:
            for item in self._loader(self.train_dataset, shuffle=True):
                loss, metrics = compute_loss(
                    item, processor=self.processor, thinker=self.thinker, avr=self.avr, cfg=self.cfg, global_step=step
                )
                (loss / self.cfg.gradient_accumulation_steps).backward()  # type: ignore[no-untyped-call]
                accum_loss += metrics.loss
                accum += 1
                if accum < self.cfg.gradient_accumulation_steps:
                    continue

                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.avr.parameters() if p.requires_grad], self.cfg.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                running_loss += accum_loss / self.cfg.gradient_accumulation_steps
                accum_loss = 0.0
                accum = 0
                step += 1

                if step % self.cfg.logging_steps == 0:
                    avg_loss = running_loss / self.cfg.logging_steps
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info("step=%d loss=%.4f lr=%.2e", step, avg_loss, lr)
                    self._log_wandb({"train/loss": avg_loss, "train/lr": lr}, step)
                    running_loss = 0.0
                if step % self.cfg.eval_steps == 0:
                    self._maybe_checkpoint(step, self.evaluate(step=step))
                if step >= self.max_steps:
                    break

        save_artifact(
            self.avr,
            self.cfg,
            self.final_dir,
            reproducibility={"global_step": step, "best_val_loss": self._best_val_loss},
        )
        logger.info("Stage 1 training complete: %d steps -> %s", step, self.final_dir)
