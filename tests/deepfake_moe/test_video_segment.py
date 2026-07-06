"""Tests for video segmentation helpers."""

import numpy as np
import pytest

from deepfake_moe.utils.video_segment import sample_frame_indices, segment_frames


def test_sample_frame_indices_spans_clip() -> None:
    idx = sample_frame_indices(total_frames=100, num_samples=5)
    assert idx.tolist() == [0, 25, 50, 74, 99]


def test_sample_frame_indices_clamps_to_total() -> None:
    idx = sample_frame_indices(total_frames=3, num_samples=10)
    assert idx.tolist() == [0, 1, 2]


def test_sample_frame_indices_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        sample_frame_indices(total_frames=0, num_samples=4)


def test_segment_frames_splits_with_short_tail() -> None:
    frames = np.zeros((7, 2, 2, 3), dtype=np.uint8)
    segments = segment_frames(frames, segment_len=3)
    assert [len(s) for s in segments] == [3, 3, 1]


def test_segment_frames_exact_division() -> None:
    frames = np.zeros((6, 2, 2, 3), dtype=np.uint8)
    segments = segment_frames(frames, segment_len=2)
    assert len(segments) == 3
    assert all(len(s) == 2 for s in segments)
