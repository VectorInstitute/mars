"""ASR stub."""

from __future__ import annotations

import numpy as np


def transcribe_audio(waveform: np.ndarray, sample_rate: int = 16_000) -> str:
    """Transcribe waveform to text (not implemented)."""
    raise NotImplementedError("transcribe_audio is a stub; add an ASR backend.")
