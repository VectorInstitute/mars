"""Optional video frame sampling and chunking (explicit opt-in)."""

from __future__ import annotations

import numpy as np


def sample_frame_indices(total_frames: int, num_samples: int) -> np.ndarray:
    """Uniformly pick frame indices across a clip.

    Examples
    --------
    >>> sample_frame_indices(100, 5).tolist()
    [0, 25, 50, 74, 99]
    """
    if total_frames <= 0 or num_samples <= 0:
        raise ValueError("total_frames and num_samples must be positive")
    num_samples = min(num_samples, total_frames)
    return np.linspace(0, total_frames - 1, num_samples).round().astype(np.int64)


def segment_frames(frames: np.ndarray, segment_len: int) -> list[np.ndarray]:
    """Split frames into fixed-length chunks (call explicitly when needed).

    Examples
    --------
    >>> import numpy as np
    >>> frames = np.zeros((6, 2, 2, 3), dtype=np.uint8)
    >>> [len(s) for s in segment_frames(frames, 2)]
    [2, 2, 2]
    """
    if segment_len <= 0:
        raise ValueError(f"segment_len must be positive, got {segment_len}")
    return [frames[i : i + segment_len] for i in range(0, len(frames), segment_len)]
