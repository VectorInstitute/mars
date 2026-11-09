"""Spliced FF++ clips: part authentic, part manipulated, with per-frame labels.

Each FF++ manipulated video XXX_YYY is made from the original XXX, frame by frame.
A spliced clip takes the frames inside a segment from the manipulated video and the
other frames from the original, at the same source positions, so only the face
changes at a cut. Some Face2Face and NeuralTextures videos are a few pixels narrower
than their original (a centre crop); the original is cropped to match.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mars.data.transforms import preprocess
from mars.data.video import read_frames


POSITIONS = ("start", "middle", "end")


@dataclass(frozen=True)
class Pair:
    """An FF++ original and a manipulated video built on it."""

    original: str
    manipulated: str
    method: str
    original_path: str
    manipulated_path: str
    n_frames: int
    frame_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class SpliceSpec:
    """One spliced clip: which pair, and which clip frames are manipulated."""

    pair: Pair
    length: int
    position: str

    def segment(self, n_frames: int) -> tuple[int, int]:
        """First and last manipulated clip frame (inclusive); (0, -1) when none."""
        if self.length == 0:
            return 0, -1
        if self.length >= n_frames:
            return 0, n_frames - 1
        start = {"start": 0, "middle": (n_frames - self.length) // 2, "end": n_frames - self.length}[self.position]
        return start, start + self.length - 1


def ffpp_test_pairs(manifest: str | Path, min_frames: int) -> list[Pair]:
    """Pairs of FF++ test originals and manipulated videos with enough shared frames.

    Parameters
    ----------
    manifest : str or Path
        FF++ manifest with video_id, path, method, label, n_frames and split columns.
    min_frames : int
        Shared source frames a pair needs (clip length times frame stride).

    Returns
    -------
    list of Pair
        One pair per manipulated test video whose original is in the test split.
    """
    with open(manifest) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    originals = {r["video_id"].split("/")[-1]: r for r in rows if r["label"] == "0"}
    pairs = []
    for r in rows:
        if r["label"] != "1":
            continue
        target = r["video_id"].split("/")[-1].split("_")[0]
        original = originals.get(target)
        if original is None:
            continue
        shared = min(int(r["n_frames"]), int(original["n_frames"]))
        if shared >= min_frames:
            pairs.append(Pair(original["video_id"], r["video_id"], r["method"], original["path"], r["path"], shared))
    return pairs


def centre_crop(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Crop (N, H, W, 3) frames to (height, width) around the centre."""
    height, width = size
    top, left = (frames.shape[1] - height) // 2, (frames.shape[2] - width) // 2
    return frames[:, top : top + height, left : left + width]


def border_difference(pair: Pair, indices: list[int], margin: float = 0.15) -> float:
    """Mean absolute pixel difference of the two videos outside the central region.

    The face sits inside the frame, so on aligned videos the border differs only by
    compression noise (about 1-3 grey levels on FF++ test); on unrelated videos it is
    above 50. The original is centre-cropped to the manipulated frame size first.
    """
    b = read_frames(pair.manipulated_path, indices).astype(np.float32)
    a = read_frames(pair.original_path, indices).astype(np.float32)
    if a.shape[1] < b.shape[1] or a.shape[2] < b.shape[2]:
        return float("inf")
    a = centre_crop(a, b.shape[1:3])
    h, w = a.shape[1:3]
    dh, dw = max(1, int(h * margin)), max(1, int(w * margin))
    border = np.ones((h, w), dtype=bool)
    border[dh : h - dh, dw : w - dw] = False
    return float(np.abs(a - b)[:, border].mean())


def aligned_pairs(
    pairs: list[Pair], indices: list[int], tolerance: float = 10.0
) -> tuple[list[Pair], list[tuple[Pair, float]]]:
    """Keep the pairs whose borders agree; record the manipulated frame size.

    Returns
    -------
    tuple
        The accepted pairs (with frame_size set) and the rejected pairs with their
        border difference.
    """
    accepted, rejected = [], []
    for pair in pairs:
        difference = border_difference(pair, indices)
        if difference <= tolerance:
            height, width = read_frames(pair.manipulated_path, [0]).shape[1:3]
            accepted.append(replace(pair, frame_size=(int(height), int(width))))
        else:
            rejected.append((pair, difference))
    return accepted, rejected


def splice_specs(
    pairs: list[Pair], lengths: tuple[int, ...], n_frames: int, positions: tuple[str, ...] = POSITIONS
) -> list[SpliceSpec]:
    """Every pair with every segment length and position, plus the unspliced references.

    Length 0 (position "none") is the clip fully from the original and length n_frames
    (position "all") the clip fully from the manipulated video; both are added once
    per pair.
    """
    specs = []
    for pair in pairs:
        specs += [SpliceSpec(pair, length, position) for length in lengths for position in positions]
        specs += [SpliceSpec(pair, 0, "none"), SpliceSpec(pair, n_frames, "all")]
    return specs


class SplicedClips(Dataset):
    """Spliced clips as (frames, valid, labels, index) samples.

    Clip frame t comes from source frame t * stride of the manipulated video when t is
    inside the segment, else of the original. labels[t] is 1 for manipulated frames.
    """

    def __init__(self, specs: list[SpliceSpec], n_frames: int, stride: int, image_size: int) -> None:
        self.specs = specs
        self.n_frames = n_frames
        self.stride = stride
        self.image_size = image_size

    def __len__(self) -> int:
        """Return the number of clips."""
        return len(self.specs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Frames (T, 3, S, S), valid (T,), labels (T,) and the clip index."""
        spec = self.specs[index]
        source = [t * self.stride for t in range(self.n_frames)]
        first, last = spec.segment(self.n_frames)
        labels = torch.zeros(self.n_frames, dtype=torch.long)
        labels[first : last + 1] = 1
        size = spec.pair.frame_size
        if size is None:
            raise ValueError("pair has no frame_size; build pairs with aligned_pairs")
        original = (
            centre_crop(read_frames(spec.pair.original_path, source), size) if bool((labels == 0).any()) else None
        )
        manipulated = read_frames(spec.pair.manipulated_path, source) if bool(labels.any()) else None
        frames = np.stack(
            [(manipulated if labels[t] else original)[t] for t in range(self.n_frames)]  # type: ignore[index]
        )
        valid = torch.ones(self.n_frames, dtype=torch.bool)
        return preprocess(frames, self.image_size), valid, labels, index
