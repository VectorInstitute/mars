"""DFD-HR on a tiny, randomly initialised CLIP."""

from __future__ import annotations

import torch

from mars.config import load_yaml, parse_overrides
from mars.models.dfd_hr import DFDHR, load_config
from mars.models.dfd_hr.model import tiles
from mars.models.dfd_hr.routing import MoEAdapter, spearman_corr
from mars.models.factory import build_detector


TINY = [
    "model.pretrained=false",
    "model.clip_size=32",
    "model.clip_patch=8",
    "model.clip_width=32",
    "model.clip_layers=4",
    "model.clip_heads=2",
    "model.clip_dim=16",
    "model.routed_blocks=[3]",
    "model.router_hidden=8",
    "model.encoder_chunk=3",
    "data.n_frames=4",
    "data.image_size=64",
]


def model_and_objective() -> tuple[DFDHR, torch.nn.Module]:
    """Build the network and objective the trainer would use."""
    detector = build_detector("dfd_hr", load_yaml("configs/models/dfd_hr/config.yaml", parse_overrides(TINY)))
    network = detector.build_network()
    assert isinstance(network, DFDHR)
    # Open every layer router so the routed block runs for each view.
    for router in network.layer_routers.values():
        torch.nn.init.constant_(router.fc2.weight, 10.0)
        torch.nn.init.constant_(router.fc1.weight, 0.1)
    return network, detector.build_objective(None)


def clips(batch: int = 2) -> torch.Tensor:
    """Random frames in [-1, 1]."""
    return torch.rand(batch, 4, 3, 64, 64) * 2 - 1


def test_shipped_config_is_the_paper_setting() -> None:
    cfg = load_config("configs/models/dfd_hr/config.yaml")
    m = cfg.model
    assert (m.routed_blocks, m.capacity, m.n_experts, m.top_k) == ([21, 22, 23], 0.75, 4, 4)
    assert cfg.data.image_size == 448 and cfg.loss.ranking == 0.1
    assert (cfg.train.micro_batch_size, cfg.train.batch_size) == (2, 32)


def test_tiles_match_the_released_chessboard_split() -> None:
    x = torch.randn(3, 2, 8, 8)
    # Released split: rearrange "b c (nh h) (nw w) -> (nh nw b) c h w".
    expected = torch.cat([x[:, :, 4 * i : 4 * i + 4, 4 * j : 4 * j + 4] for i in range(2) for j in range(2)])
    torch.testing.assert_close(tiles(x, 2), expected)


def test_moe_adapter_with_every_expert_is_the_gate_weighted_sum() -> None:
    adapter = MoEAdapter(8, n_experts=4, top_k=4, ratio=0.25)
    x = torch.randn(2, 5, 8)
    gates = torch.softmax(adapter.gate(x), dim=-1)
    expected = x + sum(gates[..., i : i + 1] * expert(x) for i, expert in enumerate(adapter.experts))
    torch.testing.assert_close(adapter(x), expected)


def test_spearman_peaks_for_the_same_order_and_carries_no_gradient() -> None:
    x = torch.randn(2, 6, requires_grad=True)
    corr = spearman_corr(x, x.detach() * 3 + 1)
    # As released: unbiased std with a mean over L, so identical ranks give (L - 1) / L.
    torch.testing.assert_close(corr, torch.full((2,), 5 / 6))
    assert not corr.requires_grad


def test_clip_and_layer_routers_are_frozen() -> None:
    model, _ = model_and_objective()
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert not any(p.requires_grad for p in model.layer_routers.parameters())
    model.train()
    assert not model.encoder.training


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
    torch.manual_seed(0)
    model, objective = model_and_objective()
    model.train()
    valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
    loss, parts = objective(model(clips(), valid=valid), torch.tensor([0, 1]))
    loss.backward()
    assert set(parts) == {"ce", "ranking"}
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing
