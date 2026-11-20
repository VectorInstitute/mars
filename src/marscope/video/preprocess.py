"""The MARS input protocol: decode, keep every second frame, cut clips, resize, scale.

This must match training exactly. Any change here changes scores without raising an
error, so every change needs the integration tests (see CONTRIBUTING.md).
"""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import av
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VideoMeta:
    """Container metadata, read before decoding."""

    duration_s: float
    width: int
    height: int
    fps: float
    size_bytes: int
    rotation: int = 0  # degrees counter-clockwise to show the video upright (phone videos); display only


# The web app's limits; the CLI warns past them (see marscope.cli) instead of refusing.
MAX_SECONDS = 120.0
MAX_LONG_SIDE = 1920
MAX_SHORT_SIDE = 1080
DAMAGED = (
    "the file looks damaged or incomplete (for example still downloading, or cut short); "
    "re-export or re-download it and try again"
)


def describe_read_error(error: Exception) -> str:
    """Turn a decoding error into a message a person can act on.

    Parameters
    ----------
    error : Exception
        What PyAV or :func:`read_meta` raised.

    Returns
    -------
    str
        The message.

    Examples
    --------
    >>> describe_read_error(ValueError("clip.mp4 has no video stream"))
    'clip.mp4 has no video stream'
    """
    if isinstance(error, av.FFmpegError):
        return DAMAGED
    return str(error)


@dataclass
class Clip:
    """One clip, preprocessed for every image size a selected model needs.

    Attributes
    ----------
    index : int
        Clip number, from 0.
    frame_indices : list of int
        Original (decoded) frame index of each real kept frame in the clip.
    n_valid : int
        Number of real frames; the rest repeat the last real frame.
    inputs : dict of int to numpy.ndarray
        Image size to an array of shape (n_frames, 3, size, size), float32 in [-1, 1].
    """

    index: int
    frame_indices: list[int]
    n_valid: int
    inputs: dict[int, np.ndarray] = field(repr=False)


def read_meta(path: str | Path) -> VideoMeta:
    """Read duration, resolution and frame rate without decoding the video.

    Parameters
    ----------
    path : str or Path
        Video file.

    Returns
    -------
    VideoMeta
        The container's metadata.

    Raises
    ------
    ValueError
        If the file has no video stream.
    """
    path = Path(path)
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{path.name} has no video stream")
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / 1_000_000
        else:
            duration = 0.0
        first = next(container.decode(stream), None)  # the display rotation is read from a frame
        rotation = int(round(getattr(first, "rotation", 0) or 0)) % 360 if first is not None else 0
        return VideoMeta(
            duration_s=duration,
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            fps=fps,
            size_bytes=path.stat().st_size,
            rotation=rotation,
        )


def iter_kept_frames(path: str | Path, stride: int) -> Iterator[tuple[int, np.ndarray]]:
    """Decode a video as RGB and yield every ``stride``-th frame.

    Parameters
    ----------
    path : str or Path
        Video file.
    stride : int
        Keep frames 0, stride, 2 * stride, ... (2 for all released models).

    Yields
    ------
    tuple of (int, numpy.ndarray)
        Decoded frame index and the frame as (H, W, 3) uint8 RGB.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"  # frame threading: same frames, faster decode
        for i, frame in enumerate(container.decode(stream)):
            if i % stride == 0:
                yield i, frame.to_ndarray(format="rgb24")


def to_input(rgb: np.ndarray, size: int) -> np.ndarray:
    """Turn one RGB frame into model input.

    Square resize without crop, bicubic with antialiasing, each channel resized as a
    float image with PIL, then scaled from [0, 1] to [-1, 1]. This is the model cards'
    reference code.

    Parameters
    ----------
    rgb : numpy.ndarray
        (H, W, 3) uint8 RGB frame.
    size : int
        Output side S (336 or 448).

    Returns
    -------
    numpy.ndarray
        (3, S, S) float32 in [-1, 1].

    Examples
    --------
    >>> to_input(np.zeros((10, 20, 3), dtype=np.uint8), 4).shape
    (3, 4, 4)
    """
    x = rgb.astype(np.float32) / 255.0
    if x.shape[:2] != (size, size):
        channels = [
            Image.fromarray(np.ascontiguousarray(x[..., c])).resize((size, size), Image.Resampling.BICUBIC)
            for c in range(3)
        ]
        x = np.clip(np.stack([np.asarray(c) for c in channels], axis=-1), 0.0, 1.0)
    return ((x - 0.5) / 0.5).transpose(2, 0, 1).astype(np.float32)


def build_clips(
    frames: Iterable[tuple[int, np.ndarray]],
    sizes: Iterable[int],
    n_frames: int,
    on_frame: Callable[[int, np.ndarray], None] | None = None,
) -> Iterator[Clip]:
    """Cut kept frames into consecutive clips and preprocess them, streaming.

    Only one clip per size is held in memory. A short last clip (or a short video) is
    padded by repeating its last real frame.

    Parameters
    ----------
    frames : iterable of (int, numpy.ndarray)
        Kept frames, as from :func:`iter_kept_frames`.
    sizes : iterable of int
        Image sizes needed by the selected models.
    n_frames : int
        Clip length (64 for all released models).
    on_frame : callable, optional
        Called with (frame index, RGB frame) for every kept frame, before resizing; used
        for face detection and thumbnails.

    Yields
    ------
    Clip
        Clips in order.

    Raises
    ------
    ValueError
        If no frame was decoded.
    """
    sizes = sorted(set(sizes))
    buffers: dict[int, list[np.ndarray]] = {s: [] for s in sizes}
    indices: list[int] = []
    clip_index = 0
    for frame_index, rgb in frames:
        if on_frame is not None:
            on_frame(frame_index, rgb)
        for s in sizes:
            buffers[s].append(to_input(rgb, s))
        indices.append(frame_index)
        if len(indices) == n_frames:
            yield _make_clip(clip_index, indices, buffers, n_frames)
            clip_index += 1
            buffers = {s: [] for s in sizes}
            indices = []
    if indices:
        yield _make_clip(clip_index, indices, buffers, n_frames)
    elif clip_index == 0:
        raise ValueError("no frames could be decoded")


def _make_clip(index: int, indices: list[int], buffers: dict[int, list[np.ndarray]], n_frames: int) -> Clip:
    n_valid = len(indices)
    inputs = {}
    for size, frames in buffers.items():
        padded = frames + [frames[-1]] * (n_frames - n_valid)
        inputs[size] = np.stack(padded).astype(np.float32)
    return Clip(index=index, frame_indices=list(indices), n_valid=n_valid, inputs=inputs)


def read_clip(  # noqa: PLR0913, PLR0917
    path: str | Path,
    clip_index: int,
    size: int,
    n_frames: int,
    stride: int,
    transform: Callable[[int, np.ndarray], np.ndarray] | None = None,
) -> Clip:
    """Decode and preprocess a single clip again (for probes and occlusion maps).

    Parameters
    ----------
    path : str or Path
        Video file.
    clip_index : int
        Clip number, from 0.
    size : int
        Image size.
    n_frames : int
        Clip length.
    stride : int
        Frame stride.
    transform : callable, optional
        Applied to each decoded RGB frame, as ``transform(frame_index, rgb)``, before
        the usual preprocessing (what-if edits).

    Returns
    -------
    Clip
        The clip, with only ``size`` in ``inputs``.

    Raises
    ------
    IndexError
        If the video has fewer clips.
    """
    start = clip_index * n_frames
    selected = ((i, rgb) for k, (i, rgb) in enumerate(iter_kept_frames(path, stride)) if start <= k < start + n_frames)
    if transform is not None:
        selected = ((i, transform(i, rgb)) for i, rgb in selected)
    try:
        for clip in build_clips(selected, [size], n_frames):
            clip.index = clip_index
            return clip
    except ValueError:
        pass
    raise IndexError(f"clip {clip_index} does not exist")


def read_clips(  # noqa: PLR0913, PLR0917
    path: str | Path,
    clip_indices: list[int],
    size: int,
    n_frames: int,
    stride: int,
    transform: Callable[[int, np.ndarray], np.ndarray] | None = None,
) -> dict[int, Clip]:
    """Decode the video once and return only the given clips.

    Parameters
    ----------
    path : str or Path
        Video file.
    clip_indices : list of int
        Clip numbers wanted.
    size : int
        Image size.
    n_frames : int
        Clip length.
    stride : int
        Frame stride.
    transform : callable, optional
        Applied to each decoded RGB frame before preprocessing (what-if edits).

    Returns
    -------
    dict of int to Clip
        The clips found.
    """
    wanted = set(clip_indices)
    last = max(wanted)
    out: dict[int, Clip] = {}
    selected = ((i, rgb) for k, (i, rgb) in enumerate(iter_kept_frames(path, stride)) if k < (last + 1) * n_frames)
    if transform is not None:
        selected = ((i, transform(i, rgb)) for i, rgb in selected)
    for clip in build_clips(selected, [size], n_frames):
        if clip.index in wanted:
            out[clip.index] = clip
        if clip.index >= last:
            break
    return out
