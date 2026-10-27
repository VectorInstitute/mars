"""Effort on a tiny, randomly initialised CLIP."""

from __future__ import annotations

import torch

from mars.config import load_yaml, parse_overrides
from mars.models.effort import Effort, load_config
from mars.models.effort.svd import SVDResidualLinear
from mars.models.factory import build_detector


TINY = [
    "model.pretrained=false",
    "model.clip_size=32",
    "model.clip_patch=8",
    "model.clip_width=32",
    "model.clip_layers=2",
    "model.clip_heads=2",
    "model.clip_dim=16",
    "model.encoder_chunk=3",
    "data.n_frames=4",
    "data.image_size=32",
]


def model_and_objective() -> tuple[Effort, torch.nn.Module]:
    """Build the network and objective the trainer would use."""
    detector = build_detector("effort", load_yaml("configs/models/effort/config.yaml", parse_overrides(TINY)))
    network = detector.build_network()
    assert isinstance(network, Effort)
    return network, detector.build_objective(None)


def clips(batch: int = 2) -> torch.Tensor:
    """Random frames in [-1, 1]."""
    return torch.rand(batch, 4, 3, 32, 32) * 2 - 1


def test_shipped_config_is_the_paper_setting() -> None:
    cfg = load_config("configs/models/effort/config.yaml")
    assert (cfg.model.residual_rank, cfg.loss.regularisation, cfg.data.image_size) == (1, 1.0, 224)


def test_split_layer_starts_as_the_pretrained_layer() -> None:
    layer = torch.nn.Linear(16, 16)
    split = SVDResidualLinear(layer.weight, layer.bias, residual_rank=1)
    x = torch.randn(3, 16)
    torch.testing.assert_close(split(x), layer(x), atol=1e-5, rtol=1e-5)
    assert split.orthogonal_loss() < 1e-4 and split.singular_value_loss() < 1e-3


def test_constraints_match_the_released_formulas() -> None:
    layer = torch.nn.Linear(8, 8)
    split = SVDResidualLinear(layer.weight, layer.bias, residual_rank=2)
    with torch.no_grad():
        split.U_residual.add_(0.3)
        split.S_residual.mul_(2)
    u = torch.cat([split.U_r, split.U_residual], dim=1)
    v = torch.cat([split.V_r, split.V_residual], dim=0)
    orthogonal = 0.5 * torch.norm(u @ u.t() - torch.eye(8), p="fro") + 0.5 * torch.norm(
        v @ v.t() - torch.eye(8), p="fro"
    )
    weight = split.weight_main + split.U_residual @ torch.diag(split.S_residual) @ split.V_residual
    keep = torch.abs(torch.norm(weight, p="fro") ** 2 - torch.norm(layer.weight, p="fro") ** 2)
    torch.testing.assert_close(split.orthogonal_loss(), orthogonal)
    torch.testing.assert_close(split.singular_value_loss(), keep)


def test_effort_at_init_equals_plain_clip() -> None:
    model, _ = model_and_objective()
    model.eval()
    pixels = torch.rand(2, 3, 32, 32) * 2 - 1
    normalised = (pixels * 0.5 + 0.5 - model.mean) / model.std
    with torch.no_grad():
        x = model.encoder.embed(normalised)
        for block in model.encoder.blocks:
            x = block(x)
        expected = model.head(model.encoder.ln_post(x[:, 0]))
        torch.testing.assert_close(model.frame_logits(pixels), expected, atol=1e-4, rtol=1e-4)


def test_only_residuals_and_head_train() -> None:
    model, _ = model_and_objective()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("head.") or n.split(".")[-1].endswith("_residual") for n in trainable)
    assert not model.state_dict().keys() & {n for n, _ in model.named_buffers() if "weight_main" in n}


def test_padding_content_never_reaches_the_logits() -> None:
    model, _ = model_and_objective()
    model.eval()
    clip = clips(1)
    other = clip.clone()
    other[:, 2:] = torch.rand(1, 2, 3, 32, 32)
    valid = torch.tensor([[True, True, False, False]])
    with torch.no_grad():
        torch.testing.assert_close(model(clip, valid=valid).logits, model(other, valid=valid).logits)


def test_loss_reaches_every_trainable_part() -> None:
    model, objective = model_and_objective()
    model.train()
    valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
    loss, parts = objective(model(clips(), valid=valid), torch.tensor([0, 1]))
    loss.backward()
    assert set(parts) == {"ce", "regularisation"}
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing
