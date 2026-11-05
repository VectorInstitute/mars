"""ReStraV with a tiny stand-in for DINOv2."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mars.config import load_yaml, parse_overrides
from mars.models.factory import build_detector
from mars.models.restrav import ReStraV, load_config
from mars.models.restrav.geometry import trajectory_features
from mars.models.restrav.model import RunningStandardiser


TINY = ["data.n_frames=10", "data.image_size=16", "model.encoder_chunk=3"]


class TinyDINO(torch.nn.Module):
    """Stand-in returning DINOv2's forward_features keys."""

    def __init__(self) -> None:
        super().__init__()
        self.patch = torch.nn.Conv2d(3, 8, kernel_size=8, stride=8)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Class and patch tokens of x."""
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        return {"x_norm_clstoken": tokens.mean(dim=1), "x_norm_patchtokens": tokens}


@pytest.fixture(autouse=True)
def tiny_dino(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace torch.hub DINOv2 with the stand-in."""
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: TinyDINO())


def released_features(z: torch.Tensor) -> torch.Tensor:
    """features_from_Z from the released dinov2_features.py, copied verbatim."""
    delta = z[:, 1:, :] - z[:, :-1, :]
    d = delta.norm(dim=-1)
    cos = F.cosine_similarity(delta[:, :-1, :], delta[:, 1:, :], dim=-1)
    t = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))

    def moment4(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return x.mean(dim=-1), x.amin(dim=-1), x.amax(dim=-1), x.var(dim=-1, unbiased=False)

    mu_d, mn_d, mx_d, var_d = moment4(d)
    mu_t, mn_t, mx_t, var_t = moment4(t)
    stats = torch.stack([mu_d, mn_d, mx_d, var_d, mu_t, mn_t, mx_t, var_t], dim=1)
    return torch.cat([d[:, :7], t[:, :6], stats], dim=1)


def model_and_objective() -> tuple[ReStraV, torch.nn.Module]:
    """Build the network and objective the trainer would use."""
    detector = build_detector("restrav", load_yaml("configs/models/restrav/config.yaml", parse_overrides(TINY)))
    network = detector.build_network()
    assert isinstance(network, ReStraV)
    return network, detector.build_objective(None)


def test_shipped_config_is_the_paper_setting() -> None:
    cfg = load_config("configs/models/restrav/config.yaml")
    assert (cfg.model.encoder_id, cfg.model.early_steps, cfg.model.early_angles) == ("dinov2_vits14", 7, 6)
    assert cfg.model.hidden == [64, 32] and cfg.data.image_size == 224


def test_features_match_the_released_code_when_every_frame_is_genuine() -> None:
    z = torch.randn(3, 24, 50)
    ours = trajectory_features(z, torch.ones(3, 24, dtype=torch.bool), 7, 6)
    torch.testing.assert_close(ours, released_features(z))


def test_features_of_a_padded_clip_equal_those_of_its_genuine_frames() -> None:
    z = torch.randn(1, 12, 50)
    valid = torch.arange(12)[None] < 9
    padded = z.clone()
    padded[:, 9:] = torch.randn(1, 3, 50)
    ours = trajectory_features(padded, valid, 7, 6)
    torch.testing.assert_close(ours, released_features(z[:, :9]))


def test_standardiser_matches_the_statistics_of_all_samples_seen() -> None:
    standardise = RunningStandardiser(4).train()
    batches = [torch.randn(5, 4) * 3 + 1 for _ in range(4)]
    for batch in batches:
        standardise(batch)
    everything = torch.cat(batches)
    torch.testing.assert_close(standardise.mean, everything.mean(dim=0))
    torch.testing.assert_close(standardise.var, everything.var(dim=0, unbiased=False))


def test_padding_content_never_reaches_the_logits() -> None:
    model, _ = model_and_objective()
    model.eval()
    clip = torch.rand(1, 10, 3, 16, 16) * 2 - 1
    other = clip.clone()
    other[:, 9:] = torch.rand(1, 1, 3, 16, 16)
    valid = torch.arange(10)[None] < 9
    with torch.no_grad():
        torch.testing.assert_close(model(clip, valid=valid).logits, model(other, valid=valid).logits)


def test_only_the_mlp_trains_and_every_part_gets_a_gradient() -> None:
    model, objective = model_and_objective()
    model.train()
    valid = torch.ones(2, 10, dtype=torch.bool)
    loss, _ = objective(model(torch.rand(2, 10, 3, 16, 16) * 2 - 1, valid=valid), torch.tensor([0, 1]))
    loss.backward()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith(("mlp.", "head.")) for n in trainable)
    assert all(p.grad is not None for n, p in model.named_parameters() if n in trainable)
