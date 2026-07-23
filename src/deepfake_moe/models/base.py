"""Shared types and ``Predictor`` protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class ClipData:
    """All inputs needed to run inference on a single clip.

    Attributes
    ----------
    clip_id : str
        Manifest clip identifier.
    dataset : str
        Source dataset name (e.g. ``lavdf``, ``av1m``).
    has_audio : bool
        Whether a usable audio stream is present.
    frames : np.ndarray
        Video frames ``(T, H, W, 3)`` uint8, already subsampled.
    waveform : np.ndarray | None
        Mono waveform ``(N,)`` float32; ``None`` when no audio.
    label_video : int
        Ground truth video label: 0 real, 1 fake.
    label_audio : int
        Ground truth audio label: 0 real, 1 fake.
    fake_periods : list[list[float]]
        Ground-truth fake intervals ``[[t0, t1], ...]``; empty if unavailable.
    video_path : str
        Resolved on-disk video path (used by vLLM eval; HF ``predict`` uses ``frames``).
    audio_path : str | None
        Resolved on-disk audio path when audio is present.
    """

    clip_id: str
    dataset: str
    has_audio: bool
    frames: np.ndarray
    waveform: np.ndarray | None
    label_video: int
    label_audio: int
    fake_periods: list[list[float]]
    video_path: str = ""
    audio_path: str | None = None


@dataclass
class PredictionOutput:
    """Structured output from any model implementing ``Predictor``.

    Matches the locked schema (``skills/02-schema-and-scoring.md``): probabilities,
    not integer confidence.

    Attributes
    ----------
    video_fake : bool
        Model video-stream prediction.
    audio_fake : bool
        Model audio-stream prediction.
    video_fake_prob : float
        Probability the video stream is manipulated, in ``[0.0, 1.0]``. AUC is
        computed from the first-token generation logprob, not this written value
        alone (see ``eval/logprob.py``); this field carries whichever is available.
    audio_fake_prob : float
        Probability the audio stream is manipulated, in ``[0.0, 1.0]``.
    fake_periods : list[list[float]]
        Predicted fake intervals ``[[t0, t1], ...]`` in seconds; empty if none.
    evidence : list[dict]
        Supporting cues; interpretability only, not scored.
    reason : str
        Short explanation from the model.
    raw_response : str
        Verbatim model output for debugging.
    parse_ok : bool
        ``False`` when JSON extraction failed.
    """

    video_fake: bool
    audio_fake: bool
    video_fake_prob: float
    audio_fake_prob: float
    fake_periods: list[list[float]]
    evidence: list[dict]
    reason: str
    raw_response: str = ""
    parse_ok: bool = True


@runtime_checkable
class Predictor(Protocol):
    """Protocol all detection models must implement.

    The eval runner accepts any object that satisfies this protocol.
    Swap the model implementation, keep the harness unchanged.
    """

    def predict(self, clip: ClipData) -> PredictionOutput:
        """Run inference on one clip and return structured predictions."""
        ...
