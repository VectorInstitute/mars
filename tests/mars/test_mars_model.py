"""Two-path MARS on a tiny, randomly initialised Perception Encoder."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from mars.config import parse_overrides
from mars.models.mars import build_model, load_config
from mars.models.mars.config import Config
from mars.models.mars.model import Mars
from mars.models.mars.semantic_path import frame_subset


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


def tiny(*overrides: str) -> Config:
    """Load the single-path config at tiny PE dimensions."""
    return load_config("tests/mars/configs/single_path.yaml", parse_overrides([*TINY, *overrides]))


def clips(batch: int = 2) -> torch.Tensor:
    """Random frames in [-1, 1]."""
    return torch.rand(batch, 4, 3, 64, 64) * 2 - 1


def full(batch: int = 2) -> torch.Tensor:
    """Mark every frame genuine."""
    return torch.ones(batch, 4, dtype=torch.bool)


@pytest.mark.parametrize("paths", ["[semantic]", "[residual,semantic]"])
def test_mars_forward_and_tuned_layernorms(paths: str) -> None:
    model = build_model(tiny(f"model.paths={paths}").model, 4, 64)
    assert isinstance(model, Mars)
    out = model(clips(), valid=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool))
    assert out.logits.shape == (2, 2)
    (out.logits.sum() + sum(path.sum() for path in out.path_logits)).backward()
    blocks = model.encoder.backbone.blocks
    assert blocks[7].norm1.weight.grad is not None
    assert not blocks[5].norm1.weight.requires_grad


def test_gate_starts_as_an_even_mix_of_standardised_margins() -> None:
    model = build_model(tiny("model.paths=[residual,semantic]").model, 4, 64).eval()
    with torch.no_grad():
        out = model(clips(), valid=full())
    margin = lambda logits: logits[:, 1] - logits[:, 0]  # noqa: E731
    semantic, residual = out.path_logits
    # Untrained running statistics (mean 0, var 1) leave each margin unchanged.
    expected = (margin(semantic) + margin(residual)) / 2 / (1 + 1e-5) ** 0.5
    torch.testing.assert_close(margin(out.logits), expected)


def test_fused_loss_reaches_only_the_gate() -> None:
    model = build_model(tiny("model.paths=[residual,semantic]").model, 4, 64)
    model(clips(), valid=full()).logits.sum().backward()
    assert model.gate.weight.grad is not None
    assert model.semantic.classifier.weight.grad is None
    assert model.residual.classifier.weight.grad is None


def test_margins_are_standardised_on_the_batch_in_training() -> None:
    model = build_model(tiny("model.paths=[residual,semantic]").model, 4, 64).train()
    out = model(clips(batch=6), valid=full(6))
    margin = out.logits[:, 1] - out.logits[:, 0]
    # Gate starts at 0.5 and both standardised margins have zero batch mean.
    assert abs(float(margin.detach().mean())) < 1e-4


def test_residual_only_reads_the_chosen_block() -> None:
    model = build_model(tiny().model, 4, 64)
    assert model.encoder.layer == 6
    assert model(clips(), valid=full()).logits.shape == (2, 2)


def test_frame_subset_stays_inside_genuine_frames() -> None:
    valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    index = frame_subset(valid, 3)
    assert index.tolist() == [[0, 2, 3], [0, 0, 1]]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("model.paths=[]", "non-empty"),
        ("model.paths=[optical_flow]", "non-empty"),
        ("model.output=semantic", "not fused or one of paths"),
    ],
)
def test_bad_path_options_are_rejected(override: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        tiny(override)


def test_semantic_path_needs_a_splittable_encoder() -> None:
    with pytest.raises(ValueError, match="can be split"):
        load_config(
            "tests/mars/configs/single_path.yaml",
            parse_overrides(["model.paths=[semantic]", "model.encoder_id=facebook/vit-large"]),
        )


def test_shipped_mars_config_uses_both_paths() -> None:
    cfg = load_config("configs/models/mars/config.yaml")
    assert cfg.model.paths == ["residual", "semantic"]
    assert (cfg.model.encoder_id, cfg.model.encoder_layer) == ("vit_pe_core_large_patch14_336.fb", 12)
    assert (cfg.model.d_feature, cfg.model.n_tokens, cfg.data.image_size) == (1024, 576, 336)
    assert cfg.model.semantic_early_norms


def test_semantic_layernorms_stay_frozen_when_tuning_is_off() -> None:
    model = build_model(tiny("model.paths=[residual,semantic]", "model.tune_semantic_norms=false").model, 4, 64)
    assert not any(p.requires_grad for p in model.encoder.parameters())


def test_split_after_the_last_block_leaves_the_semantic_path_only_the_head() -> None:
    depth = len(build_model(tiny("model.paths=[semantic]").model, 4, 64).encoder.backbone.blocks)
    model = build_model(tiny("model.paths=[residual,semantic]", f"model.encoder_layer={depth}").model, 4, 64)
    assert model(clips(), valid=full()).logits.shape == (2, 2)


def test_semantic_lora_off_matches_mars_bitwise() -> None:
    cfg = tiny("model.paths=[residual,semantic]")
    torch.manual_seed(0)
    plain = build_model(cfg.model, 4, 64)
    torch.manual_seed(0)
    zero = build_model(tiny("model.paths=[residual,semantic]", "model.semantic_lora_rank=0").model, 4, 64)
    for (name, a), (_, b) in zip(plain.state_dict().items(), zero.state_dict().items(), strict=True):
        assert torch.equal(a, b), name


def test_semantic_lora_starts_as_mars_and_leaves_its_init_untouched() -> None:
    torch.manual_seed(0)
    plain = build_model(tiny("model.paths=[residual,semantic]").model, 4, 64).eval()
    torch.manual_seed(0)
    lora = build_model(tiny("model.paths=[residual,semantic]", "model.semantic_lora_rank=4").model, 4, 64).eval()
    ours = {k.replace(".base.", "."): v for k, v in lora.state_dict().items() if "lora_" not in k}
    assert ours.keys() == plain.state_dict().keys()
    for name, value in plain.state_dict().items():
        assert torch.equal(value, ours[name]), name
    x, valid = clips(), full()
    a, b = plain(x, valid=valid), lora(x, valid=valid)
    assert torch.equal(a.logits, b.logits)


def test_semantic_lora_trains_only_the_semantic_blocks() -> None:
    model = build_model(tiny("model.paths=[residual,semantic]", "model.semantic_lora_rank=4").model, 4, 64)
    blocks = model.encoder.backbone.blocks
    split = model.split
    out = model(clips(), valid=full())
    sum(path.sum() for path in out.path_logits).backward()
    for index, block in enumerate(blocks):
        adapted = [n for n, _ in block.named_parameters() if "lora_" in n]
        assert len(adapted) == (8 if index >= split else 0)
    assert blocks[split].attn.qkv.lora_b.grad is not None
    assert not blocks[split].attn.qkv.base.weight.requires_grad
    assert not any(p.requires_grad for p in blocks[:split].parameters())


def test_semantic_lora_needs_the_semantic_path() -> None:
    with pytest.raises(ValueError, match="semantic_lora_rank needs"):
        tiny("model.paths=[residual]", "model.semantic_lora_rank=4")


def test_shipped_semantic_lora_config() -> None:
    cfg = load_config("configs/models/mars/semantic_lora.yaml")
    assert (cfg.model.semantic_lora_rank, cfg.model.semantic_lora_alpha) == (16, 8.0)
    assert cfg.model.paths == ["residual", "semantic"] and cfg.model.encoder_layer == 12


def test_semantic_early_norms_off_adds_nothing() -> None:
    torch.manual_seed(0)
    plain = build_model(tiny("model.paths=[residual,semantic]").model, 4, 64)
    torch.manual_seed(0)
    off = build_model(tiny("model.paths=[residual,semantic]", "model.semantic_early_norms=false").model, 4, 64)
    assert plain.early_norms is None and off.early_norms is None
    for (name, a), (_, b) in zip(plain.state_dict().items(), off.state_dict().items(), strict=True):
        assert torch.equal(a, b), name


def test_semantic_early_norms_start_as_mars() -> None:
    torch.manual_seed(0)
    plain = build_model(tiny("model.paths=[residual,semantic]").model, 4, 64).eval()
    torch.manual_seed(0)
    early = build_model(tiny("model.paths=[residual,semantic]", "model.semantic_early_norms=true").model, 4, 64).eval()
    shared = {k: v for k, v in early.state_dict().items() if not k.startswith("early_norms.")}
    assert shared.keys() == plain.state_dict().keys()
    for name, value in plain.state_dict().items():
        assert torch.equal(value, shared[name]), name
    # norm_pre plus norm1 and norm2 of each block before the split
    assert len(early.early_norms) == 1 + 2 * early.split
    x, valid = clips(), torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    with torch.no_grad():
        a, b = plain(x, valid=valid), early(x, valid=valid)
    for p, q in zip(a.path_logits, b.path_logits, strict=True):
        torch.testing.assert_close(p, q)
    torch.testing.assert_close(a.logits, b.logits)


def test_semantic_early_norms_train_the_copies_not_the_shared_blocks() -> None:
    model = build_model(tiny("model.paths=[residual,semantic]", "model.semantic_early_norms=true").model, 4, 64)
    semantic, residual = model(clips(), valid=full()).path_logits
    semantic.sum().backward()
    blocks = model.encoder.backbone.blocks
    assert model.early_norms["0_norm1"].weight.grad is not None
    assert model.early_norms["norm_pre"].weight.grad is not None
    assert not blocks[0].norm1.weight.requires_grad and blocks[0].norm1.weight.grad is None
    assert blocks[model.split].norm1.weight.grad is not None
    assert model.residual.classifier.weight.grad is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (["model.paths=[residual]", "model.semantic_early_norms=true"], "needs the semantic path"),
        (
            ["model.paths=[semantic]", "model.semantic_early_norms=true", "model.tune_semantic_norms=false"],
            "needs tune",
        ),
    ],
)
def test_bad_semantic_early_norms_options_are_rejected(overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        tiny(*overrides)


def test_semantic_lora_and_early_norms_train_together() -> None:
    model = build_model(
        tiny("model.paths=[residual,semantic]", "model.semantic_early_norms=true", "model.semantic_lora_rank=4").model,
        4,
        64,
    )
    semantic, _ = model(clips(), valid=full()).path_logits
    semantic.sum().backward()
    blocks = model.encoder.backbone.blocks
    assert model.early_norms["0_norm1"].weight.grad is not None
    assert blocks[model.split].attn.qkv.lora_b.grad is not None
    assert not any(p.requires_grad for p in blocks[: model.split].parameters())
