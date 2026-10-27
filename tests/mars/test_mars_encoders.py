"""SigLIP and DINOv3 split for two-path MARS, on tiny randomly initialised models."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from mars.config import parse_overrides
from mars.models.mars import build_model, load_config
from mars.models.mars.encoder import FrozenDINOv3, FrozenSigLIP
from mars.models.mars.model import Mars


def tiny_siglip() -> Any:
    from transformers import SiglipVisionConfig, SiglipVisionModel

    cfg = SiglipVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=2, image_size=64, patch_size=16
    )
    return SiglipVisionModel(cfg)


def tiny_dinov3() -> Any:
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    cfg = DINOv3ViTConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        image_size=64,
        patch_size=16,
        num_register_tokens=4,
    )
    return DINOv3ViTModel(cfg)


@pytest.fixture(autouse=True)
def tiny_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return tiny random models instead of downloading weights."""
    import transformers

    torch.manual_seed(0)
    monkeypatch.setattr(transformers.SiglipVisionModel, "from_pretrained", staticmethod(lambda *a, **k: tiny_siglip()))
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", staticmethod(lambda *a, **k: tiny_dinov3()))


def pixels() -> torch.Tensor:
    return torch.rand(2, 3, 64, 64) * 2 - 1


def encoders() -> list[Any]:
    return [
        FrozenSigLIP("google/siglip", None, 4, d_feature=32, n_tokens=16),
        FrozenDINOv3("facebook/dinov3", None, 4, d_feature=32, n_tokens=16),
    ]


@pytest.mark.parametrize("index", [0, 1])
def test_split_pass_matches_the_full_model(index: int) -> None:
    encoder = encoders()[index]
    x = pixels()
    with torch.no_grad():
        hidden, context = encoder.hidden(x, 2)
        pooled = encoder.finish(hidden, context, 2)
        full_input = encoder.normalize(x) if isinstance(encoder, FrozenDINOv3) else x
        full = encoder.backbone(pixel_values=full_input)
    torch.testing.assert_close(pooled, full.pooler_output)
    assert encoder.tokens(x).shape == (2, 16, 32)


@pytest.mark.parametrize("index", [0, 1])
def test_only_late_layernorms_train(index: int) -> None:
    encoder = encoders()[index]
    encoder.tune_layernorms(2)
    trainable = {name for name, p in encoder.backbone.named_parameters() if p.requires_grad}
    blocks = "encoder.layers" if index == 0 else "model.layer"
    assert trainable, "no LayerNorm was made trainable"
    assert not any(name.startswith(f"{blocks}.0.") or name.startswith(f"{blocks}.1.") for name in trainable)
    assert any(name.startswith(f"{blocks}.3.") for name in trainable)
    assert not any(name.startswith("embeddings") for name in trainable)


TINY = [
    "model.d_feature=32",
    "model.n_tokens=16",
    "model.grid_side=4",
    "model.d_model=32",
    "model.d_resid=8",
    "model.depth=2",
    "model.n_heads=2",
    "model.pool_hidden=8",
    "model.grad_checkpoint=false",
    "model.encoder_layer=2",
    "model.semantic_frames=3",
    "data.n_frames=4",
    "data.image_size=64",
]


@pytest.mark.parametrize("encoder_id", ["google/siglip-so400m-patch14-384", "facebook/dinov3-vitl16-pretrain-lvd1689m"])
def test_two_path_mars_runs_on_the_encoder(encoder_id: str) -> None:
    cfg = load_config("configs/models/mars/config.yaml", parse_overrides([*TINY, f"model.encoder_id={encoder_id}"]))
    model = build_model(cfg.model, 4, 64)
    assert isinstance(model, Mars)
    out = model(torch.rand(2, 4, 3, 64, 64) * 2 - 1, valid=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool))
    assert out.logits.shape == (2, 2)
    sum(path.sum() for path in out.path_logits).backward()
    late = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
    assert late and all(p.grad is not None for p in late)


def test_shipped_backbone_configs() -> None:
    siglip = load_config("configs/models/mars/ablation_backbone_siglip.yaml")
    assert (siglip.model.d_feature, siglip.model.n_tokens, siglip.model.grid_side, siglip.model.encoder_layer) == (
        1152,
        729,
        27,
        13,
    )
    dinov3 = load_config("configs/models/mars/ablation_backbone_dinov3.yaml")
    assert (dinov3.model.d_feature, dinov3.model.n_tokens, dinov3.model.grid_side, dinov3.model.encoder_layer) == (
        1024,
        576,
        24,
        12,
    )
    assert siglip.model.paths == dinov3.model.paths == ["residual", "semantic"]


@pytest.mark.parametrize("encoder_id", ["google/siglip-so400m-patch14-384", "facebook/dinov3-vitl16-pretrain-lvd1689m"])
def test_early_norms_start_as_mars_and_train_only_the_copies(encoder_id: str) -> None:
    overrides = [*TINY, f"model.encoder_id={encoder_id}"]
    torch.manual_seed(1)
    plain = build_model(
        load_config(
            "configs/models/mars/config.yaml", parse_overrides([*overrides, "model.semantic_early_norms=false"])
        ).model,
        4,
        64,
    )
    torch.manual_seed(1)
    early = build_model(
        load_config(
            "configs/models/mars/config.yaml", parse_overrides([*overrides, "model.semantic_early_norms=true"])
        ).model,
        4,
        64,
    )
    # one copy per LayerNorm of the blocks before the split; no pre-norm outside PE
    assert len(early.early_norms) == 2 * early.split
    x, valid = torch.rand(2, 4, 3, 64, 64) * 2 - 1, torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    plain.eval(), early.eval()
    with torch.no_grad():
        a, b = plain(x, valid=valid), early(x, valid=valid)
    for p, q in zip(a.path_logits, b.path_logits, strict=True):
        torch.testing.assert_close(p, q)
    early.train()
    semantic, _ = early(x, valid=valid).path_logits
    semantic.sum().backward()
    assert all(p.grad is not None for p in early.early_norms.parameters())
    assert all(p.grad is None for p in early.encoder.layers[0].parameters())
