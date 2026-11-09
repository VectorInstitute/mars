"""ForAda on a tiny, randomly initialised CLIP and adapter."""

from __future__ import annotations

import torch

from mars.config import load_yaml, parse_overrides
from mars.models.factory import build_detector
from mars.models.forada import ForAda, load_config
from mars.models.forada.model import query_mask


TINY = [
    "model.pretrained=false",
    "model.clip_size=32",
    "model.clip_patch=8",
    "model.clip_width=32",
    "model.clip_layers=4",
    "model.clip_heads=2",
    "model.clip_dim=16",
    "model.adapter_blocks=3",
    "model.n_queries=8",
    "model.fusion_map={1: 1, 2: 2, 3: 3}",
    "model.intra_block=2",
    "model.mlp_dim=16",
    "model.mlp_out_dim=8",
    "model.encoder_chunk=3",
    "data.n_frames=4",
    "data.image_size=64",
]


def model_and_objective() -> tuple[ForAda, torch.nn.Module]:
    """Build the network and objective the trainer would use."""
    detector = build_detector("forada", load_yaml("configs/models/forada/config.yaml", parse_overrides(TINY)))
    network = detector.build_network()
    assert isinstance(network, ForAda)
    return network, detector.build_objective(None)


def clips(batch: int = 2) -> torch.Tensor:
    """Random frames in [-1, 1]."""
    return torch.rand(batch, 4, 3, 64, 64) * 2 - 1


def test_shipped_config_is_the_paper_setting() -> None:
    cfg = load_config("configs/models/forada/config.yaml")
    m = cfg.model
    assert (m.n_queries, m.adapter_blocks, m.fusion_map, m.intra_block) == (128, 8, {1: 1, 2: 8, 3: 15}, 7)
    assert cfg.data.image_size == 256
    assert cfg.log.group == "main_ffpp_gtav"


def test_clip_is_frozen_and_the_rest_trains() -> None:
    model, _ = model_and_objective()
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for name, p in model.named_parameters() if not name.startswith("encoder."))
    model.train()
    assert not model.encoder.training


def test_queries_see_only_themselves_and_the_patches() -> None:
    bias = torch.randn(1, 2, 3, 2, 2)
    mask = query_mask(bias, n_tokens=5)
    assert mask.shape == (1, 2, 8, 8)
    assert mask[0, 0, 0, 0] == 0 and mask[0, 0, 0, 1] == -100
    assert mask[0, 0, 0, 3] == -100
    torch.testing.assert_close(mask[0, :, :3, 4:], bias.flatten(3)[0])
    assert (mask[0, :, 3:, :3] == -100).all() and (mask[0, :, 3:, 3:] == 0).all()


def test_clip_logit_is_the_mean_over_genuine_frames() -> None:
    model, _ = model_and_objective()
    model.eval()
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    with torch.no_grad():
        out = model(clips(), valid=valid)
    assert out.logits.shape == (2, 2)
    torch.testing.assert_close(out.logits[1], out.frame_logits[1, :2].mean(dim=0))


def test_padding_content_never_reaches_the_logits() -> None:
    model, _ = model_and_objective()
    model.eval()
    clip = clips(1)
    other = clip.clone()
    other[:, 2:] = torch.rand(1, 2, 3, 64, 64)
    valid = torch.tensor([[True, True, False, False]])
    with torch.no_grad():
        torch.testing.assert_close(model(clip, valid=valid).logits, model(other, valid=valid).logits)


def test_loss_reaches_every_trainable_part() -> None:
    model, objective = model_and_objective()
    model.train()
    valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
    loss, parts = objective(model(clips(), valid=valid), torch.tensor([0, 1]))
    loss.backward()
    assert set(parts) == {"ce"}
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing
    assert all(p.grad is None for p in model.encoder.parameters())
