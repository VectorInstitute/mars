"""Optional audio windowing (explicit opt-in)."""

from __future__ import annotations

import numpy as np


def window_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    window_seconds: float,
    hop_seconds: float | None = None,
    *,
    drop_last: bool = False,
) -> list[np.ndarray]:
    """Slice a mono waveform into windows (call explicitly when needed).

    Examples
    --------
    >>> import numpy as np
    >>> w = np.arange(2500, dtype=np.float32)
    >>> [len(x) for x in window_waveform(w, 1000, 1.0)]
    [1000, 1000, 500]
    """
    if waveform.ndim != 1:
        raise ValueError(f"expected a 1-D mono waveform, got shape {waveform.shape}")
    if window_seconds <= 0:
        raise ValueError(f"window_seconds must be positive, got {window_seconds}")

    hop_seconds = window_seconds if hop_seconds is None else hop_seconds
    if hop_seconds <= 0:
        raise ValueError(f"hop_seconds must be positive, got {hop_seconds}")

    window = int(round(window_seconds * sample_rate))
    hop = int(round(hop_seconds * sample_rate))

    windows: list[np.ndarray] = []
    for start in range(0, len(waveform), hop):
        chunk = waveform[start : start + window]
        if len(chunk) == 0 or (len(chunk) < window and drop_last):
            break
        windows.append(chunk)
    return windows
