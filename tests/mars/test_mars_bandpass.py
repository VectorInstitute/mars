"""Band-pass filters and the validity helpers behind MARS."""

from __future__ import annotations

import pytest
import torch

from mars.models.mars.bandpass import (
    gaussian_matrix,
    heat_matrix,
    prefix_tables,
    scale_pairs,
    smooth_grid,
    smooth_time,
    smooth_time_masked,
    smoothing_matrix,
)
from mars.models.mars.validity import is_prefix, observation_index


FRAMES, TOKENS, DIM, SIDE = 12, 16, 8, 4


@pytest.mark.parametrize("kind", ["dog", "heat"])
def test_smoothing_preserves_constants(kind: str) -> None:
    matrix = smoothing_matrix(FRAMES, 1.0, kind)
    assert torch.allclose(matrix.sum(dim=1), torch.ones(FRAMES, dtype=torch.float64), atol=1e-9)


@pytest.mark.parametrize("kind", ["dog", "heat"])
def test_band_annihilates_the_constant_mode(kind: str) -> None:
    low = smoothing_matrix(FRAMES, 0.5, kind)
    high = smoothing_matrix(FRAMES, 2.0, kind)
    constant = torch.ones(FRAMES, dtype=torch.float64)
    assert torch.allclose(low @ constant - high @ constant, torch.zeros(FRAMES, dtype=torch.float64), atol=1e-9)


def test_heat_response_is_bounded_by_one() -> None:
    values = torch.linalg.eigvalsh(heat_matrix(FRAMES, 1.0))
    assert float(values.max()) <= 1.0 + 1e-9


def test_gaussian_rejects_a_non_positive_width() -> None:
    with pytest.raises(ValueError, match="sigma must be positive"):
        gaussian_matrix(FRAMES, 0.0)


def test_scale_pairs_makes_one_band_per_consecutive_pair() -> None:
    assert scale_pairs([0.5, 1.0, 2.0, 4.0]) == [(0.5, 1.0), (1.0, 2.0), (2.0, 4.0)]


@pytest.mark.parametrize(
    ("scales", "message"),
    [([1.0], "at least two"), ([2.0, 1.0], "strictly increasing"), ([0.0, 1.0], "must be positive")],
)
def test_scale_pairs_rejects_a_bad_ladder(scales: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        scale_pairs(scales)


def test_grid_smoothing_is_separable_and_symmetric() -> None:
    x = torch.randn(2, 3, SIDE * SIDE, DIM)
    matrix = smoothing_matrix(SIDE, 1.0, "dog").float()
    out = smooth_grid(x, matrix, SIDE)
    assert out.shape == x.shape
    transposed = smooth_grid(x.reshape(2, 3, SIDE, SIDE, DIM).transpose(2, 3).reshape(2, 3, -1, DIM), matrix, SIDE)
    assert torch.allclose(
        out.reshape(2, 3, SIDE, SIDE, DIM).transpose(2, 3),
        transposed.reshape(2, 3, SIDE, SIDE, DIM),
        atol=1e-5,
    )


def test_grid_smoothing_rejects_a_mismatched_grid() -> None:
    with pytest.raises(ValueError, match="do not form"):
        smooth_grid(torch.randn(1, 1, 15, DIM), smoothing_matrix(SIDE, 1.0, "dog").float(), SIDE)


def test_prefix_table_row_matches_the_dense_matrix() -> None:
    table = prefix_tables(FRAMES, 1.0, "dog")
    assert table.shape == (FRAMES + 1, FRAMES, FRAMES)
    assert torch.allclose(table[FRAMES], smoothing_matrix(FRAMES, 1.0, "dog"), atol=1e-12)
    assert torch.allclose(table[5, :5, :5], smoothing_matrix(5, 1.0, "dog"), atol=1e-12)


def test_prefix_table_rows_still_sum_to_one() -> None:
    table = prefix_tables(FRAMES, 1.0, "dog")
    for length in range(2, FRAMES + 1):
        rows = table[length].sum(dim=1)
        assert torch.allclose(rows, torch.ones(FRAMES, dtype=torch.float64), atol=1e-9)


def test_masked_smoothing_never_mixes_across_the_padding_edge() -> None:
    table = prefix_tables(FRAMES, 1.0, "dog").float()
    x = torch.zeros(1, FRAMES, TOKENS, DIM)
    x[:, 6:] = 100.0  # everything past the valid prefix
    valid = torch.arange(FRAMES)[None, :] < 6
    out = smooth_time_masked(x, table, valid)
    assert torch.allclose(out[:, :6], torch.zeros_like(out[:, :6]), atol=1e-6)


def test_masked_smoothing_equals_dense_when_nothing_is_padded() -> None:
    table = prefix_tables(FRAMES, 1.0, "dog").float()
    x = torch.randn(2, FRAMES, TOKENS, DIM)
    valid = torch.ones(2, FRAMES, dtype=torch.bool)
    dense = smooth_time(x, smoothing_matrix(FRAMES, 1.0, "dog").float())
    assert torch.allclose(smooth_time_masked(x, table, valid), dense, atol=1e-6)


def test_is_prefix_accepts_a_prefix_and_rejects_a_gap() -> None:
    assert is_prefix(torch.tensor([[True, True, False], [True, False, False]]))
    assert not is_prefix(torch.tensor([[True, False, True]]))


def test_observation_index_counts_up_for_a_prefix() -> None:
    valid = torch.tensor([[True, True, True, False, False]])
    assert torch.equal(observation_index(valid), torch.tensor([[0, 1, 2, 2, 2]]))
