"""Frame sampling: keep every ``frame_stride``-th frame, then cut into clips."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Clip:
    """One sample: ``n_frames`` source-frame indices plus provenance."""

    video_id: str
    clip_index: int
    frame_indices: tuple[int, ...]
    n_valid: int

    @property
    def is_padded(self) -> bool:
        """True when the clip was padded to ``n_frames``."""
        return self.n_valid < len(self.frame_indices)


def sampled_indices(n_source_frames: int, frame_stride: int = 2) -> list[int]:
    """Frames kept from a video: 0, stride, 2*stride, ..."""
    if n_source_frames <= 0:
        return []
    return list(range(0, n_source_frames, frame_stride))


def segment(
    indices: list[int],
    n_frames: int = 64,
    clip_stride: int | None = None,
    padding: str = "last_frame",
) -> list[tuple[tuple[int, ...], int]]:
    """Cut ``indices`` into fixed-length segments.

    Returns ``(frame_indices, n_valid)`` pairs. ``clip_stride`` defaults to
    ``n_frames`` (non-overlapping). A video shorter than one segment still
    yields exactly one padded segment.
    """
    if not indices:
        return []
    stride = clip_stride or n_frames
    out: list[tuple[tuple[int, ...], int]] = []
    for start in range(0, max(len(indices), 1), stride):
        window = indices[start : start + n_frames]
        if not window:
            break
        n_valid = len(window)
        if n_valid < n_frames:
            if padding == "last_frame":
                pad_value = window[-1]
            elif padding == "zeros":
                pad_value = -1
            else:
                raise ValueError(f"unknown padding '{padding}'")
            window = window + [pad_value] * (n_frames - n_valid)
        out.append((tuple(window), n_valid))
        if start + n_frames >= len(indices):
            break
    return out


def clips_for_video(
    video_id: str,
    n_source_frames: int,
    n_frames: int = 64,
    frame_stride: int = 2,
    clip_stride: int | None = None,
    padding: str = "last_frame",
) -> list[Clip]:
    """Full sampling pipeline for one video."""
    kept = sampled_indices(n_source_frames, frame_stride)
    segments = segment(kept, n_frames, clip_stride, padding)
    return [
        Clip(video_id=video_id, clip_index=i, frame_indices=idx, n_valid=n_valid)
        for i, (idx, n_valid) in enumerate(segments)
    ]
