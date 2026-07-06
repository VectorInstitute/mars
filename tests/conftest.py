"""Shared test fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_rate() -> int:
    """Return a small, round sample rate for fast waveform tests."""
    return 1_000


@pytest.fixture
def sine_waveform(sample_rate: int) -> np.ndarray:
    """Return a 2.5-second mono sine wave (deterministic, 1-D float32)."""
    t = np.arange(int(2.5 * sample_rate)) / sample_rate
    return np.sin(2 * np.pi * 5.0 * t).astype(np.float32)
