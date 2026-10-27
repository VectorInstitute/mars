"""DeMamba on a tiny, randomly initialised X-CLIP, and the X-CLIP tower itself."""

from __future__ import annotations

import torch
from transformers import XCLIPVisionConfig, XCLIPVisionModel

from mars.config import load_yaml, parse_overrides
from mars.models.demamba import DeMamba, load_config
from mars.models.demamba.mamba import MambaBlock
from mars.models.demamba.model import snake_order
from mars.models.demamba.xclip import XCLIPVisionTransformer
from mars.models.factory import build_detector


TINY = [
    "model.pretrained=false",
    "model.patch_size=8",
    "model.width=16",
    "model.layers=2",
    "model.heads=2",
    "model.hidden=32",
    "model.message_group=2",
    "model.d_state=4",
    "data.n_frames=4",
    "data.image_size=32",
]


def model_and_objective(*overrides: str) -> tuple[DeMamba, torch.nn.Module]:
    """Build the network and objective the trainer would use."""
    raw = load_yaml("configs/models/demamba/config.yaml", parse_overrides([*TINY, *overrides]))
    detector = build_detector("demamba", raw)
    network = detector.build_network()
    assert isinstance(network, DeMamba)
    return network, detector.build_objective(None)


def clips(batch: int = 2) -> torch.Tensor:
    """Random frames in [-1, 1]."""
    return torch.rand(batch, 4, 3, 32, 32) * 2 - 1


def test_shipped_config_is_the_paper_setting() -> None:
    cfg = load_config("configs/models/demamba/config.yaml")
    assert cfg.model.encoder_id == "microsoft/xclip-base-patch16"
    assert (cfg.model.zone, cfg.model.message_group, cfg.data.image_size) == (1, 8, 224)


def test_xclip_tower_matches_hugging_face_without_padding() -> None:
    config = XCLIPVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        image_size=32, patch_size=8, num_frames=2,
    )  # fmt: skip
    released = XCLIPVisionModel(config).eval()
    tower = XCLIPVisionTransformer(32, 8, 16, 2, 2, 32, group=2).eval()
    tower.load_state_dict(released.state_dict())
    frames = torch.randn(2, 4, 3, 32, 32)
    with torch.no_grad():
        tokens, pooled = tower(frames, torch.ones(2, 4, dtype=torch.bool))
        expected = released(pixel_values=frames.flatten(0, 1))
    torch.testing.assert_close(tokens.flatten(0, 1), expected.last_hidden_state, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(pooled.flatten(0, 1), expected.pooler_output, atol=1e-5, rtol=1e-5)


def test_snake_order_matches_the_released_reorder() -> None:
    assert snake_order(3).tolist() == [0, 3, 6, 7, 4, 1, 2, 5, 8]


def test_zone_of_one_patch_is_that_patch_across_frames() -> None:
    model, _ = model_and_objective()
    patches = torch.randn(2, 4, 16, 16)
    sequences = model.zone_sequences(patches)
    assert sequences.shape == (2 * 16, 4, 16)
    torch.testing.assert_close(sequences[16 + 5], patches[1, :, 5])


def test_mamba_outputs_never_read_later_steps() -> None:
    block = MambaBlock(8, d_state=4, expand=2, d_conv=4).eval()
    x = torch.randn(2, 6, 8)
    changed = x.clone()
    changed[:, 4:] = torch.randn(2, 2, 8)
    with torch.no_grad():
        torch.testing.assert_close(block(x)[:, :4], block(changed)[:, :4])


def test_padding_content_never_reaches_the_logits() -> None:
    for zone in ("model.zone=1", "model.zone=2"):
        model, _ = model_and_objective(zone)
        model.eval()
        clip = clips(1)
        other = clip.clone()
        other[:, 3:] = torch.rand(1, 1, 3, 32, 32)
        valid = torch.tensor([[True, True, True, False]])
        with torch.no_grad():
            torch.testing.assert_close(model(clip, valid=valid).logits, model(other, valid=valid).logits)


def test_loss_reaches_every_parameter() -> None:
    model, objective = model_and_objective()
    model.train()
    valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
    loss, parts = objective(model(clips(), valid=valid), torch.tensor([0, 1]))
    loss.backward()
    assert set(parts) == {"ce"}
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing
