"""GenD on a tiny, randomly initialised Perception Encoder."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from mars.config import load_yaml, parse_overrides
from mars.models.factory import build_detector
from mars.models.gend import GenD, load_config
from mars.models.gend.config import Config
from mars.models.gend.objective import alignment, uniformity


TINY = [
    "model.encoder_id=vit_pe_core_tiny_patch16_384",
    "model.d_feature=192",
    "model.n_tokens=16",
    "model.encoder_chunk=3",
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


def test_shipped_config_is_the_paper_setting() -> None:
    cfg: Config = load_config("configs/models/gend/config.yaml")
    assert (cfg.loss.ce, cfg.loss.uniformity, cfg.loss.alignment) == (1.0, 0.5, 0.1)
    assert cfg.model.encoder_id.startswith("vit_pe_core_large")
    assert cfg.log.group == "main_ffpp_gtav"


def model_and_objective() -> tuple[GenD, torch.nn.Module]:
    """Build the network and objective the trainer would use."""
    detector = build_detector("gend", load_yaml("configs/models/gend/config.yaml", parse_overrides(TINY)))
    network = detector.build_network()
    assert isinstance(network, GenD)
    return network, detector.build_objective(None)


def clips(batch: int = 2) -> torch.Tensor:
    """Random frames in [-1, 1]."""
    return torch.rand(batch, 4, 3, 64, 64) * 2 - 1


def test_only_norms_and_head_are_trainable() -> None:
    model, _ = model_and_objective()
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all("norm" in name or name.startswith("classifier.") for name in trainable)
    assert any(name.startswith("encoder.") for name in trainable)


def test_clip_logit_is_the_mean_over_genuine_frames() -> None:
    model, _ = model_and_objective()
    model.eval()
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    with torch.no_grad():
        out = model(clips(), valid=valid)
    assert out.logits.shape == (2, 2)
    torch.testing.assert_close(out.logits[1], out.frame_logits[1, :2].mean(dim=0))
    assert out.frame_logits[1, 2:].abs().sum() == 0


def test_padding_content_never_reaches_the_logits() -> None:
    model, _ = model_and_objective()
    model.eval()
    clip = clips(1)
    other = clip.clone()
    other[:, 2:] = torch.rand(1, 2, 3, 64, 64)
    valid = torch.tensor([[True, True, False, False]])
    with torch.no_grad():
        torch.testing.assert_close(model(clip, valid=valid).logits, model(other, valid=valid).logits)


def test_loss_reaches_only_trainable_parameters() -> None:
    model, objective = model_and_objective()
    model.train()
    valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
    loss, parts = objective(model(clips(), valid=valid), torch.tensor([0, 1]))
    loss.backward()
    assert set(parts) == {"ce", "uniformity", "alignment"}
    assert model.classifier.weight.grad is not None
    assert any(p.grad is not None for n, p in model.encoder.named_parameters() if "norm" in n)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_uniformity_and_alignment_match_the_reference_formulas() -> None:
    x = torch.nn.functional.normalize(torch.randn(7, 5), dim=1)
    labels = torch.tensor([0, 0, 1, 1, 1, 0, 1])
    reference_uniformity = torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().clamp(min=1e-6).log()
    pairs = torch.nonzero((labels[:, None] == labels[None, :]).triu(diagonal=1))
    reference_alignment = (x[pairs[:, 0]] - x[pairs[:, 1]]).norm(p=2, dim=1).pow(2).mean()
    torch.testing.assert_close(uniformity(x), reference_uniformity)
    torch.testing.assert_close(alignment(x, labels), reference_alignment)
