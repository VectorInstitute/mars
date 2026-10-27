"""Masked mean over genuine frames."""

from __future__ import annotations

import pytest
import torch

from mars.models.factory import build_detector
from mars.utils.masking import masked_mean


def test_masked_mean_ignores_padded_frames() -> None:
    logits = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    valid = torch.tensor([[True, True, False]])
    torch.testing.assert_close(masked_mean(logits, valid, dim=1), torch.tensor([[2.0, 3.0]]))


def test_masked_mean_with_every_frame_valid_is_the_plain_mean() -> None:
    x = torch.randn(2, 5, 3)
    torch.testing.assert_close(masked_mean(x, torch.ones(2, 5, dtype=torch.bool), dim=1), x.mean(dim=1))


def test_unknown_model_names_the_registered_ones() -> None:
    with pytest.raises(ValueError, match="registered"):
        build_detector("nope", {})
