"""Tests for audio segmentation helpers."""

import numpy as np
import pytest

from deepfake_moe.utils.audio_segment import window_waveform


def test_window_waveform_contiguous(sine_waveform: np.ndarray, sample_rate: int) -> None:
    windows = window_waveform(sine_waveform, sample_rate, window_seconds=1.0)
    # 2.5s clip -> 1s + 1s + 0.5s tail.
    assert [len(w) for w in windows] == [1000, 1000, 500]


def test_window_waveform_drop_last(sine_waveform: np.ndarray, sample_rate: int) -> None:
    windows = window_waveform(sine_waveform, sample_rate, window_seconds=1.0, drop_last=True)
    assert [len(w) for w in windows] == [1000, 1000]


def test_window_waveform_overlap(sine_waveform: np.ndarray, sample_rate: int) -> None:
    windows = window_waveform(sine_waveform, sample_rate, window_seconds=1.0, hop_seconds=0.5)
    assert len(windows) == 5
    assert len(windows[0]) == 1000


def test_window_waveform_rejects_2d(sample_rate: int) -> None:
    with pytest.raises(ValueError):
        window_waveform(np.zeros((2, 10)), sample_rate, window_seconds=1.0)
