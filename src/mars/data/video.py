"""Video and frame-directory decoding shared by every dataset.

Clips read consecutive frames, so video files seek once to the first needed
frame and decode forward rather than seeking per frame. A directory of
numbered images is treated as one video.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def list_frame_files(path: str | Path) -> list[Path]:
    """Return numbered frame files in a directory, sorted by integer stem."""
    directory = Path(path)
    files = [entry for entry in directory.iterdir() if entry.is_file() and entry.stem.isdigit()]
    return sorted(files, key=lambda entry: int(entry.stem))


def _count_video_frames(path: Path) -> int:
    import av

    try:
        with av.open(str(path)) as container:
            return sum(1 for _ in container.decode(video=0))
    except Exception:
        return 0


def count_frames(path: str | Path) -> int:
    """Frame count for a video file or a directory of numbered images."""
    target = Path(path)
    if target.is_dir():
        return len(list_frame_files(target))
    return _count_video_frames(target)


def _read_image(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _stack_decoded(
    indices: list[int],
    decoded: dict[int, np.ndarray],
    path: Path,
) -> np.ndarray:
    if not decoded:
        raise RuntimeError(f"decoded no frames from {path}")
    fallback = decoded[max(decoded)]
    height, width, _ = fallback.shape
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    return np.stack([blank if i < 0 else decoded.get(i, fallback) for i in indices], axis=0)


def _read_frame_dir(directory: Path, indices: list[int]) -> np.ndarray:
    files = list_frame_files(directory)
    wanted = sorted({i for i in indices if i >= 0})
    decoded: dict[int, np.ndarray] = {}
    n_files = len(files)
    for index in wanted:
        if 0 <= index < n_files:
            decoded[index] = _read_image(files[index])
    return _stack_decoded(indices, decoded, directory)


def _read_video_frames(path: Path, indices: list[int]) -> np.ndarray:
    import av

    wanted = sorted({i for i in indices if i >= 0})
    decoded: dict[int, np.ndarray] = {}
    if wanted:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            target = set(wanted)
            last = wanted[-1]
            for position, frame in enumerate(container.decode(video=0)):
                if position in target:
                    decoded[position] = frame.to_ndarray(format="rgb24")
                if position >= last:
                    break
    return _stack_decoded(indices, decoded, path)


def read_frames(path: str | Path, indices: list[int]) -> np.ndarray:
    """Decode requested frame indices as uint8 RGB of shape (N, H, W, 3).

    path may be a video file or a directory of numbered images. indices may
    repeat (last-frame padding) and may contain -1 (zero padding). Frames are
    decoded once and reused for repeats.
    """
    target = Path(path)
    if target.is_dir():
        return _read_frame_dir(target, indices)
    return _read_video_frames(target, indices)
