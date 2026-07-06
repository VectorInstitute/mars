"""Basic media I/O via PyAV."""

from __future__ import annotations

import av
import numpy as np


def load_audio(path: str, sample_rate: int = 16_000) -> np.ndarray:
    """Load a file as mono float32 waveform at ``sample_rate``.

    Returns an empty array when no audio stream is present.
    """
    container = av.open(path)
    try:
        if not container.streams.audio:
            return np.array([], dtype=np.float32)
        resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=sample_rate)
        chunks: list[np.ndarray] = []
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
    finally:
        container.close()

    return np.concatenate(chunks).astype(np.float32) if chunks else np.array([], dtype=np.float32)


def extract_audio(video_path: str, sample_rate: int = 16_000) -> np.ndarray:
    """Extract the audio track from a video as a mono waveform."""
    return load_audio(video_path, sample_rate=sample_rate)


def load_video_frames(path: str, max_frames: int | None = None) -> np.ndarray:
    """Decode video frames to uint8 RGB array ``(T, H, W, 3)``.

    Decodes the full clip unless ``max_frames`` is set. For subsampling, call
    :func:`deepfake_moe.utils.video_segment.sample_frame_indices` explicitly.
    """
    container = av.open(path)
    frames: list[np.ndarray] = []
    try:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        container.close()

    return np.stack(frames).astype(np.uint8) if frames else np.empty((0, 0, 0, 3), dtype=np.uint8)
