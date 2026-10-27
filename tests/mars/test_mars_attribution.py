"""Exact frame and patch contributions of two-path MARS, on a tiny random model."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from mars.config import parse_overrides
from mars.models.mars import build_model, load_config
from mars.models.mars.attribution import attribute


TINY = [
    "model.encoder_id=vit_pe_core_tiny_patch16_384",
    "model.d_feature=192",
    "model.n_tokens=16",
    "model.grid_side=4",
    "model.d_model=32",
    "model.d_resid=8",
    "model.depth=2",
    "model.n_heads=2",
    "model.pool_hidden=8",
    "model.grad_checkpoint=false",
    "model.encoder_layer=6",
    "model.semantic_frames=3",
    "model.paths=[residual,semantic]",
    "data.n_frames=4",
    "data.image_size=64",
]


@pytest.fixture(autouse=True)
def tiny_random_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build the tiny PE at 64 px without downloading weights."""
    import timm

    create = timm.create_model

    def random_tiny(name: str, **kwargs: Any) -> torch.nn.Module:
        kwargs.pop("pretrained_cfg_overlay", None)
        return create(name, pretrained=False, img_size=64)

    monkeypatch.setattr(timm, "create_model", random_tiny)


def model(*overrides: str) -> torch.nn.Module:
    """Tiny two-path MARS in eval mode with non-trivial margin statistics and gate."""
    cfg = load_config("tests/mars/configs/single_path.yaml", parse_overrides([*TINY, *overrides]))
    torch.manual_seed(0)
    net = build_model(cfg.model, 4, 64).eval()
    with torch.no_grad():
        for norm, (mean, var) in zip(net.margin_norms, ((0.3, 2.0), (-0.5, 0.7)), strict=True):
            norm.running_mean.fill_(mean)
            norm.running_var.fill_(var)
        net.gate.weight.normal_(0, 0.5)
        net.gate.bias.fill_(0.2)
    return net


VALID = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)


@pytest.mark.parametrize("early_norms", ["true", "false"])
def test_parts_sum_to_the_fused_margin(early_norms: str) -> None:
    net = model(f"model.semantic_early_norms={early_norms}")
    torch.manual_seed(1)
    out = attribute(net, torch.rand(2, 4, 3, 64, 64) * 2 - 1, VALID)
    torch.testing.assert_close(out.margin_from_parts, out.margin, atol=1e-5, rtol=1e-5)
    assert out.frame_scores.shape == (2, 4) and out.frame_maps.shape == (2, 4, 16)
    torch.testing.assert_close(out.patch_weights.sum(dim=1), torch.ones(2))


def test_padded_frames_contribute_nothing() -> None:
    torch.manual_seed(1)
    out = attribute(model(), torch.rand(2, 4, 3, 64, 64) * 2 - 1, VALID)
    assert torch.all(out.residual_terms[1, 2:] == 0)
    assert torch.all(out.frame_scores[1, 2:] == 0)
    # the semantic path samples only genuine frames
    assert int(out.semantic_index[1].max()) <= 1


def test_attribute_needs_eval_mode_and_two_paths() -> None:
    with pytest.raises(ValueError, match="eval mode"):
        attribute(model().train(), torch.zeros(2, 4, 3, 64, 64), VALID)
    cfg = load_config("tests/mars/configs/single_path.yaml", parse_overrides([*TINY, "model.paths=[semantic]"]))
    with pytest.raises(ValueError, match="two-path"):
        attribute(build_model(cfg.model, 4, 64).eval(), torch.zeros(2, 4, 3, 64, 64), VALID)
