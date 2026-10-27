"""Spliced clips: segments, labels, alignment and centre crop, on synthetic videos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mars.eval.faithfulness.splice import (
    Pair,
    SplicedClips,
    SpliceSpec,
    aligned_pairs,
    splice_specs,
)


def write_video(folder: Path, frames: np.ndarray) -> str:
    """Store (N, H, W, 3) uint8 frames as numbered PNGs, a video the decoder reads."""
    folder.mkdir()
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(folder / f"{index:04d}.png")
    return str(folder)


@pytest.fixture()
def pair(tmp_path: Path) -> Pair:
    """Return an original and a narrower manipulated copy whose centre differs."""
    rng = np.random.default_rng(0)
    original = rng.integers(0, 256, size=(16, 20, 26, 3), dtype=np.uint8)
    manipulated = original[:, :, 1:25].copy()  # centre crop, 2 pixels narrower
    manipulated[:, 8:12, 10:14] = 255 - manipulated[:, 8:12, 10:14]  # the 'face' changes
    return Pair(
        "youtube/000",
        "Deepfakes/000_001",
        "Deepfakes",
        write_video(tmp_path / "a", original),
        write_video(tmp_path / "b", manipulated),
        16,
    )


def test_segments_cover_the_requested_frames() -> None:
    p = Pair("a", "b", "m", "", "", 0)
    assert SpliceSpec(p, 4, "start").segment(8) == (0, 3)
    assert SpliceSpec(p, 4, "middle").segment(8) == (2, 5)
    assert SpliceSpec(p, 4, "end").segment(8) == (4, 7)
    assert SpliceSpec(p, 0, "none").segment(8) == (0, -1)
    assert SpliceSpec(p, 8, "all").segment(8) == (0, 7)


def test_alignment_accepts_a_centre_crop_and_rejects_unrelated_videos(pair: Pair, tmp_path: Path) -> None:
    accepted, rejected = aligned_pairs([pair], [0, 5, 10])
    assert len(accepted) == 1 and not rejected
    assert accepted[0].frame_size == (20, 24)
    rng = np.random.default_rng(1)
    other = write_video(tmp_path / "c", rng.integers(0, 256, size=(16, 20, 24, 3), dtype=np.uint8))
    unrelated = Pair("youtube/000", "Deepfakes/000_002", "Deepfakes", pair.original_path, other, 16)
    accepted, rejected = aligned_pairs([unrelated], [0, 5, 10])
    assert not accepted and len(rejected) == 1


def test_spliced_clip_takes_the_segment_from_the_manipulated_video(pair: Pair) -> None:
    (aligned,), _ = aligned_pairs([pair], [0])
    specs = splice_specs([aligned], (2,), n_frames=8)
    assert len(specs) == 3 + 2
    clips = SplicedClips(specs, n_frames=8, stride=2, image_size=24)
    middle = specs.index(SpliceSpec(aligned, 2, "middle"))
    frames, valid, labels, index = clips[middle]
    assert labels.tolist() == [0, 0, 0, 1, 1, 0, 0, 0] and bool(valid.all()) and index == middle
    full_original = clips[specs.index(SpliceSpec(aligned, 0, "none"))][0]
    full_manipulated = clips[specs.index(SpliceSpec(aligned, 8, "all"))][0]
    for t in range(8):
        source = full_manipulated if labels[t] else full_original
        assert np.array_equal(frames[t].numpy(), source[t].numpy())
