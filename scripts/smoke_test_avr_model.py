"""Throwaway real-checkpoint diagnostic for models/avr/. Not part of the pipeline.

Loads the real Qwen3-Omni checkpoint, runs one clip through the frozen
encoders, then through AVRModel in both inference mode (no masking) and
Stage-1 mode (complementary masking), checking shapes and gradient flow.

Usage:
    uv run python scripts/smoke_test_avr_model.py --video clip.mp4 --audio clip.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import Qwen3OmniMoeProcessor  # noqa: E402

from deepfake_moe.models.avr.config import AVRConfig  # noqa: E402
from deepfake_moe.models.avr.encoders import encode, load_frozen_encoders  # noqa: E402
from deepfake_moe.models.avr.masking import (  # noqa: E402
    audio_token_timestamps,
    complementary_window_mask,
    visual_token_timestamps,
)
from deepfake_moe.models.avr.model import AVRModel  # noqa: E402
from deepfake_moe.models.config import Config  # noqa: E402
from deepfake_moe.models.registry import load_config  # noqa: E402
from deepfake_moe.utils.media_io import load_audio, load_video_frames, video_native_fps  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test AVRModel against a real Qwen3-Omni checkpoint.")
    parser.add_argument("--video", required=True, help="Path to a video file.")
    parser.add_argument("--audio", required=True, help="Path to an audio file.")
    parser.add_argument("--model", default="qwen3-omni-hf", help="Model name in configs/models.yaml.")
    parser.add_argument("--models-config", default="configs/models.yaml", help="Model registry YAML.")
    return parser.parse_args()


def main() -> None:
    """Build a real processor batch, encode it, and run AVRModel end to end."""
    args = _parse_args()
    config = load_config(args.model, args.models_config)
    if not isinstance(config, Config):
        raise SystemExit(f"{args.model!r} must be an hf-backend model")

    print(f"Loading processor + frozen encoders: {config.model_id}")
    processor = Qwen3OmniMoeProcessor.from_pretrained(config.model_id)
    thinker = load_frozen_encoders(config.model_id)
    # device_map="auto" can, in principle, place audio_tower/visual on
    # different devices, so each modality's inputs go to its own tower's
    # device rather than assuming a single shared device.
    audio_device = next(thinker.audio_tower.parameters()).device
    visual_device = next(thinker.visual.parameters()).device

    frames = load_video_frames(args.video, config.video_max_frames)
    waveform = load_audio(args.audio, config.audio_sample_rate)
    fps = video_native_fps(args.video)
    print(f"  video native fps: {fps:.2f} (passed to processor for correct token timestamps)")
    batch = processor(
        text=["<|im_start|>user\ndescribe<|im_end|>\n"],
        audio=[waveform],
        videos=[frames],
        fps=fps,
        return_tensors="pt",
        padding=True,
    )

    print("Running frozen encoders...")
    outputs = encode(
        thinker,
        input_features=batch["input_features"].to(audio_device),
        feature_attention_mask=batch["feature_attention_mask"].to(audio_device),
        pixel_values_videos=batch["pixel_values_videos"].to(visual_device),
        video_grid_thw=batch["video_grid_thw"].to(visual_device),
    )
    print(f"  audio_hidden: {tuple(outputs.audio_hidden.shape)}")
    print(f"  video_hidden: {tuple(outputs.video_hidden.shape)}")

    # AVRModel needs both streams on one device; audio_device is as good as any.
    device = audio_device
    audio_hidden = outputs.audio_hidden.to(device)
    video_hidden = outputs.video_hidden.to(device)

    audio_ts = audio_token_timestamps(audio_hidden.shape[0], device=device)
    visual_ts = visual_token_timestamps(
        batch["video_grid_thw"].cpu(),
        batch["video_second_per_grid"],
        thinker.visual.spatial_merge_size,
    ).to(device)

    # train/val sample caps are required by AVRConfig but unused here — this
    # script never loads a dataset.
    avr_config = AVRConfig(model_id=config.model_id, train_samples_per_dataset=1, val_samples_per_dataset=1)
    avr = AVRModel(avr_config).to(device=device, dtype=torch.bfloat16)

    print("\nRunning AVRModel (inference mode, no masking)...")
    result = avr(audio_hidden, video_hidden, audio_ts, visual_ts)
    print(f"  evidence_audio : {tuple(result.evidence_audio.shape)}")
    print(f"  evidence_visual: {tuple(result.evidence_visual.shape)}")
    print(f"  aux_loss       : {result.aux_loss.item():.4f}")

    print("\nRunning AVRModel (Stage 1 mode, complementary masking)...")
    audio_mask, visual_mask = complementary_window_mask(audio_ts, visual_ts, avr_config.mask_window_seconds)
    result_masked = avr(audio_hidden, video_hidden, audio_ts, visual_ts, audio_mask, visual_mask)
    print(f"  masked audio tokens : {audio_mask.sum().item()} / {audio_mask.numel()}")
    print(f"  masked visual tokens: {visual_mask.sum().item()} / {visual_mask.numel()}")

    print("\nChecking backward pass...")
    loss = (
        result_masked.mu_star.float().pow(2).mean()
        + result_masked.evidence_audio.float().pow(2).mean()
        + result_masked.evidence_visual.float().pow(2).mean()
        + result_masked.aux_loss
    )
    loss.backward()
    n_grad = sum(1 for p in avr.parameters() if p.grad is not None)
    n_total = sum(1 for _ in avr.parameters())
    print(f"  params with gradients: {n_grad} / {n_total} (expected: {n_total})")

    print("\nDone. Review the shapes/counts above against the known trace.")


if __name__ == "__main__":
    main()
