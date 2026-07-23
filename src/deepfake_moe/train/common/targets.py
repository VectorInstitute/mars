"""Deterministic, ground-truth-only locked-schema SFT targets."""

from __future__ import annotations

import json

from deepfake_moe.models.base import ClipData


def _normalize_periods(periods: object) -> list[list[float]]:
    if not isinstance(periods, list):
        return []
    normalized: list[list[float]] = []
    for period in periods:
        if not isinstance(period, (list, tuple)) or len(period) != 2:
            continue
        try:
            start, end = float(period[0]), float(period[1])
        except (TypeError, ValueError):
            continue
        if start < end:
            normalized.append([start, end])
    return normalized


def render_reason(
    video_fake: bool,
    audio_fake: bool,
    fake_periods: list[list[float]],
    *,
    has_audio: bool = True,
) -> str:
    """Render a non-speculative two-sentence explanation from labels and timing."""
    if not video_fake and not audio_fake:
        if not has_audio:
            return "The video stream is authentic. No audio stream or manipulated time period is present."
        return "The video stream and audio stream are both authentic. No manipulated time period is present."

    if video_fake and not has_audio:
        stream_sentence = "The video stream is manipulated and no audio stream is present."
    elif video_fake and audio_fake:
        stream_sentence = "Both the video stream and audio stream are manipulated."
    elif video_fake:
        stream_sentence = "The video stream is manipulated while the audio stream is authentic."
    else:
        stream_sentence = "The audio stream is manipulated while the video stream is authentic."

    if fake_periods:
        spans = ", ".join(f"{start:.1f}-{end:.1f}s" for start, end in fake_periods)
        timing_sentence = f"The manipulation is localized to {spans}."
    else:
        timing_sentence = "The manipulation applies across the clip."
    return f"{stream_sentence} {timing_sentence}"


def build_target(clip: ClipData) -> dict:
    """Build the locked JSON target without inventing interpretability evidence."""
    video_fake = clip.label_video == 1
    audio_fake = clip.has_audio and clip.label_audio == 1
    periods = _normalize_periods(clip.fake_periods) if video_fake or audio_fake else []
    return {
        "video_fake": video_fake,
        "audio_fake": audio_fake,
        "video_fake_prob": float(video_fake),
        "audio_fake_prob": float(audio_fake),
        "fake_periods": periods,
        "evidence": [],
        "reason": render_reason(video_fake, audio_fake, periods, has_audio=clip.has_audio),
    }


def target_json(clip: ClipData) -> str:
    """Serialize a deterministic compact assistant response."""
    return json.dumps(build_target(clip), ensure_ascii=False, separators=(",", ":"))
