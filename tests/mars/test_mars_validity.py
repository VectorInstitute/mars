"""The masking contract: padded frames must not reach any temporal computation."""

from __future__ import annotations

import pytest
import torch

from mars.config import parse_overrides
from mars.models.mars import build_model, load_config
from mars.models.mars.config import Config
from mars.models.mars.streams import ResidualStreams


CONFIG = "tests/mars/configs/single_path.yaml"
SMALL = [
    "model.encoder_id=stub",
    "model.d_feature=32",
    "model.n_tokens=16",
    "model.grid_side=4",
    "model.d_model=32",
    "model.d_resid=8",
    "model.depth=2",
    "model.n_heads=2",
    "model.pool_hidden=8",
    "model.grad_checkpoint=false",
    "data.n_frames=12",
    "data.image_size=64",
]
FRAMES, TOKENS, DIM = 12, 16, 8


def small(*overrides: str) -> Config:
    """Load the shipped MARS config at test dimensions."""
    return load_config(CONFIG, parse_overrides([*SMALL, *overrides]))


def prefix(*lengths: int) -> torch.Tensor:
    """Build validity masks with the given genuine-frame counts."""
    return torch.arange(FRAMES)[None, :] < torch.tensor(lengths)[:, None]


def time_mixing(cfg: Config) -> slice:
    """Locate the streams that mix frames: temporal bands, then the first difference."""
    n_spatial = len(cfg.model.streams.spatial.bands)
    return slice(n_spatial, cfg.model.streams.n_residual)


def padded_clip(length: int) -> torch.Tensor:
    """Build one clip whose frames past ``length`` repeat the last genuine frame."""
    x = torch.randn(1, FRAMES, TOKENS, DIM)
    x[:, length:] = x[:, length - 1 : length]
    return x


def test_padding_content_cannot_change_a_valid_residual() -> None:
    cfg = small()
    streams = ResidualStreams(cfg.model, FRAMES)
    valid = prefix(6)
    a = padded_clip(6)
    b = a.clone()
    b[:, 6:] = 1000.0  # same genuine frames, different rubbish in the padding
    window = time_mixing(cfg)
    for left, right in zip(streams(a, valid)[window], streams(b, valid)[window]):
        assert torch.allclose(left[:, :6], right[:, :6], atol=1e-6)


def test_unmasked_streams_are_contaminated_by_padding() -> None:
    """Show the guard above is meaningful: the unmasked path really does mix."""
    cfg = small("model.temporal_validity=false")
    streams = ResidualStreams(cfg.model, FRAMES)
    a = padded_clip(6)
    b = a.clone()
    b[:, 6:] = 1000.0
    window = time_mixing(cfg)
    differs = [
        not torch.allclose(left[:, :6], right[:, :6], atol=1e-6)
        for left, right in zip(streams(a, None)[window], streams(b, None)[window])
    ]
    assert any(differs)


def test_how_much_padding_there_is_does_not_matter() -> None:
    """Five genuine frames give the same residuals whether 7 slots follow or 3."""
    cfg = small()
    x = padded_clip(5)
    window = time_mixing(cfg)
    long = ResidualStreams(cfg.model, FRAMES)(x, prefix(5))[window]
    short = ResidualStreams(cfg.model, 8)(x[:, :8], prefix(5)[:, :8])[window]
    for left, right in zip(long, short):
        assert torch.allclose(left[:, :5], right[:, :5], atol=1e-6)


def test_an_unpadded_clip_matches_the_unmasked_path() -> None:
    masked = ResidualStreams(small().model, FRAMES)
    plain = ResidualStreams(small("model.temporal_validity=false").model, FRAMES)
    x = torch.randn(2, FRAMES, TOKENS, DIM)
    for left, right in zip(masked(x, torch.ones(2, FRAMES, dtype=torch.bool)), plain(x, None)):
        assert torch.allclose(left, right, atol=1e-6)


def test_a_constant_clip_has_no_temporal_response() -> None:
    cfg = small()
    streams = ResidualStreams(cfg.model, FRAMES)
    x = torch.ones(1, FRAMES, TOKENS, DIM)
    for stream in streams(x, prefix(7))[time_mixing(cfg)]:
        assert torch.allclose(stream[:, :7], torch.zeros_like(stream[:, :7]), atol=1e-6)


@pytest.mark.parametrize("length", [2, 3, 12])
def test_short_prefixes_stay_finite(length: int) -> None:
    cfg = small()
    streams = ResidualStreams(cfg.model, FRAMES)
    for stream in streams(padded_clip(length), prefix(length)):
        assert torch.isfinite(stream).all()


def test_no_gradient_flows_from_padded_inputs() -> None:
    cfg = small()
    streams = ResidualStreams(cfg.model, FRAMES)
    x = padded_clip(5).requires_grad_(True)
    total = sum(s[:, :5].sum() for s in streams(x, prefix(5))[time_mixing(cfg)])
    total.backward()
    assert x.grad is not None
    assert torch.allclose(x.grad[:, 5:], torch.zeros_like(x.grad[:, 5:]), atol=1e-8)


def test_a_ragged_batch_matches_scoring_each_clip_alone() -> None:
    cfg = small()
    streams = ResidualStreams(cfg.model, FRAMES)
    a, b = padded_clip(4), padded_clip(9)
    batch = torch.cat([a, b])
    together = streams(batch, prefix(4, 9))
    alone_a = streams(a, prefix(4))
    alone_b = streams(b, prefix(9))
    for joint, left, right in zip(together, alone_a, alone_b):
        assert torch.allclose(joint[0:1], left, atol=1e-6)
        assert torch.allclose(joint[1:2], right, atol=1e-6)


def test_the_temporal_readout_excludes_padded_frames() -> None:
    """Scoring a padded clip equals scoring only its genuine frames."""
    cfg = small()
    padded = build_model(cfg.model, FRAMES, 64).eval()
    trimmed = build_model(cfg.model, 5, 64).eval()
    trimmed.load_state_dict(padded.state_dict(), strict=True)
    clip = torch.randn(1, FRAMES, 3, 64, 64)
    clip[:, 5:] = clip[:, 4:5]
    with torch.no_grad():
        a = padded(clip, valid=prefix(5)).logits
        b = trimmed(clip[:, :5], valid=torch.ones(1, 5, dtype=torch.bool)).logits
    assert torch.allclose(a, b, atol=1e-5)


def test_a_wrong_mask_changes_a_padded_clip() -> None:
    model = build_model(small().model, FRAMES, 64).eval()
    clip = torch.randn(1, FRAMES, 3, 64, 64)
    clip[:, 4:] = clip[:, 3:4]
    with torch.no_grad():
        right = model(clip, valid=prefix(4)).logits
        wrong = model(clip, valid=torch.ones(1, FRAMES, dtype=torch.bool)).logits
    assert not torch.allclose(right, wrong, atol=1e-6)


def test_a_non_prefix_mask_is_rejected() -> None:
    model = build_model(small().model, FRAMES, 64).eval()
    mask = torch.ones(1, FRAMES, dtype=torch.bool)
    mask[0, 3] = False
    with pytest.raises(ValueError, match="contiguous prefix"):
        model(torch.randn(1, FRAMES, 3, 64, 64), valid=mask)


def test_switching_validity_off_ignores_a_supplied_mask() -> None:
    model = build_model(small("model.temporal_validity=false").model, FRAMES, 64).eval()
    clip = torch.randn(1, FRAMES, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(model(clip, valid=prefix(4)).logits, model(clip, valid=None).logits)
