"""Timestamp-based complementary masking for AVR self-supervised pretraining."""

from __future__ import annotations

import torch


# The installed audio encoder (three stride-2 Conv2d layers) turns a full
# 100-mel-frame chunk into exactly 13 output tokens (100 -> 50 -> 25 -> 13,
# via the standard conv output-length formula), and the feature extractor
# feeding it runs at 100 mel frames/sec (16kHz audio, hop_length=160). So the
# audio encoder runs at 13 Hz (~76.9ms/token), not the commonly-cited
# 12.5Hz/80ms figure. Re-derive this if the audio encoder architecture or its
# chunk-window config changes.
AUDIO_TOKENS_PER_SECOND = 13.0


def audio_token_timestamps(num_tokens: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Per-token timestamps (seconds) for one sample's audio token stream.

    Assumes uniform spacing at ``AUDIO_TOKENS_PER_SECOND`` — exact for every
    full 100-mel-frame chunk; a clip's final, possibly-partial chunk may
    deviate slightly from perfectly uniform spacing, which this
    approximation ignores as immaterial for window-bucketed masking.
    """
    return torch.arange(num_tokens, dtype=torch.float32, device=device) / AUDIO_TOKENS_PER_SECOND


def visual_token_timestamps(
    video_grid_thw: torch.Tensor,
    video_second_per_grid: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Per-token timestamps (seconds) for the spatially-merged video token stream.

    Temporal grid index ``t`` (0-indexed) has timestamp ``t *
    video_second_per_grid``; every one of the ``llm_grid_h * llm_grid_w``
    spatially-merged tokens at that index shares that timestamp
    (``llm_grid_h/w = grid_h/w // spatial_merge_size``). Concatenates across
    videos in ``video_grid_thw``'s row order, matching the pooled encoder
    output's token order.
    """
    all_timestamps = []
    for video_idx in range(video_grid_thw.shape[0]):
        grid_t, grid_h, grid_w = video_grid_thw[video_idx].tolist()
        llm_h = grid_h // spatial_merge_size
        llm_w = grid_w // spatial_merge_size
        per_frame_timestamp = torch.arange(grid_t, dtype=torch.float32) * video_second_per_grid[video_idx].float()
        all_timestamps.append(per_frame_timestamp.repeat_interleave(llm_h * llm_w))
    return torch.cat(all_timestamps)


def complementary_window_mask(
    audio_timestamps: torch.Tensor,
    visual_timestamps: torch.Tensor,
    window_seconds: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask audio in even-indexed time windows, visual in odd-indexed ones.

    At any instant exactly one modality is masked and the other is visible,
    so cross-modal imputation always has real context to reconstruct from.
    Returns ``(audio_mask, visual_mask)`` boolean tensors — ``True`` marks a
    masked (to-be-imputed) token.
    """
    audio_window_idx = torch.div(audio_timestamps, window_seconds, rounding_mode="floor").long()
    visual_window_idx = torch.div(visual_timestamps, window_seconds, rounding_mode="floor").long()
    audio_mask = audio_window_idx % 2 == 0
    visual_mask = visual_window_idx % 2 == 1
    return audio_mask, visual_mask
