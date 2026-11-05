"""MARS network wiring, stream selection, and the detector factory."""

from __future__ import annotations

import pytest
import torch

from mars.config import parse_overrides
from mars.models.factory import build_detector
from mars.models.mars import build_model, load_config
from mars.models.mars.config import Config
from mars.models.mars.encoder import (
    ENCODERS,
    FrozenDINOv3,
    FrozenPerceptionEncoder,
    FrozenSigLIP,
    StubEncoder,
    build_encoder,
)


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
    "data.n_frames=8",
    "data.image_size=64",
]


def small(*overrides: str) -> Config:
    """Load the shipped MARS config at test dimensions."""
    return load_config(CONFIG, parse_overrides([*SMALL, *overrides]))


def frames(batch: int = 2) -> torch.Tensor:
    """Draw a clip batch at the test dimensions."""
    return torch.randn(batch, 8, 3, 64, 64)


def prefix(*lengths: int) -> torch.Tensor:
    """Build validity masks with the given genuine-frame counts."""
    return torch.arange(8)[None, :] < torch.tensor(lengths)[:, None]


def test_single_path_config_builds_seven_residual_streams() -> None:
    cfg = small()
    assert cfg.model.streams.residual_names == [
        "spatial_0.5-1",
        "spatial_1-2",
        "spatial_2-4",
        "temporal_0.5-1",
        "temporal_1-2",
        "temporal_2-4",
        "first_difference",
    ]


def test_single_path_config_uses_no_block_norm() -> None:
    model = build_model(small().model, 8, 64)
    assert isinstance(model.block_norm, torch.nn.Identity)
    assert len(model.fusion.stream_projs) == 7


def test_forward_returns_logits_and_patch_weights() -> None:
    cfg = small()
    model = build_model(cfg.model, 8, 64).eval()
    out = model(frames(), valid=prefix(8, 5))
    assert out.logits.shape == (2, 2)
    assert out.patch_weights is not None and out.patch_weights.shape == (2, 16)
    assert out.pooled is not None and out.pooled.shape == (2, cfg.model.d_model)


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ("model.streams.spatial.enabled=false", 4),
        ("model.streams.temporal.enabled=false", 4),
        ("model.streams.temporal.scales=[0.5,1.0,2.0]", 6),
        ("model.streams.first_difference=false", 6),
    ],
)
def test_disabling_a_family_changes_the_stream_count(override: str, expected: int) -> None:
    cfg = small(override)
    model = build_model(cfg.model, 8, 64).eval()
    assert cfg.model.streams.n_residual == expected
    assert len(model.fusion.stream_projs) == expected
    assert model(frames(), valid=prefix(8, 5)).logits.shape == (2, 2)


def test_appearance_can_be_switched_off() -> None:
    model = build_model(small("model.streams.appearance=false").model, 8, 64).eval()
    assert model.fusion.appearance_proj is None
    assert len(model.fusion.stream_projs) == 7
    assert model(frames(), valid=prefix(8, 5)).logits.shape == (2, 2)


def test_residual_streams_can_all_be_switched_off() -> None:
    model = build_model(
        small(
            "model.streams.spatial.enabled=false",
            "model.streams.temporal.enabled=false",
            "model.streams.first_difference=false",
        ).model,
        8,
        64,
    ).eval()
    assert model.fusion.streams is None
    assert model.fusion.appearance_proj is not None
    assert model(frames(), valid=prefix(8, 5)).logits.shape == (2, 2)


def test_an_empty_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one stream"):
        small(
            "model.streams.appearance=false",
            "model.streams.spatial.enabled=false",
            "model.streams.temporal.enabled=false",
            "model.streams.first_difference=false",
        )


def test_unknown_operator_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be dog or heat"):
        small("model.streams.operator=wavelet")


def test_tokens_must_match_the_grid() -> None:
    with pytest.raises(ValueError, match="grid_side"):
        small("model.n_tokens=15")


def test_heat_and_gaussian_families_differ_under_shared_weights() -> None:
    dog = build_model(small().model, 8, 64).eval()
    heat = build_model(small("model.streams.operator=heat").model, 8, 64).eval()
    heat.load_state_dict(dog.state_dict(), strict=True)
    clip, valid = frames(), prefix(8, 5)
    with torch.no_grad():
        assert not torch.allclose(dog(clip, valid=valid).logits, heat(clip, valid=valid).logits)


def test_filter_buffers_stay_out_of_the_checkpoint() -> None:
    state = build_model(small().model, 8, 64).state_dict()
    assert not [k for k in state if ".space_" in k or ".time_" in k]


def test_trainable_parameters_exclude_the_encoder() -> None:
    model = build_model(small().model, 8, 64)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder = sum(p.numel() for p in model.encoder.parameters())
    assert encoder > 0
    assert trainable == sum(p.numel() for p in model.parameters()) - encoder


@pytest.mark.parametrize(
    ("path", "encoder", "image_size", "dims"),
    [
        ("configs/models/mars/ablation_backbone_siglip.yaml", "google/siglip-so400m-patch14-384", 384, (1152, 729, 27)),
        (
            "configs/models/mars/ablation_backbone_dinov3.yaml",
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
            384,
            (1024, 576, 24),
        ),
    ],
)
def test_backbone_ablations_change_only_the_encoder(
    path: str, encoder: str, image_size: int, dims: tuple[int, int, int]
) -> None:
    base, cfg = load_config("configs/models/mars/config.yaml"), load_config(path)
    assert (cfg.model.encoder_id, cfg.data.image_size) == (encoder, image_size)
    assert (cfg.model.d_feature, cfg.model.n_tokens, cfg.model.grid_side) == dims
    assert cfg.data.test == base.data.test  # shared.yaml still reached through the chain
    assert cfg.model.paths == base.model.paths
    assert cfg.model.streams == base.model.streams


def test_build_encoder_maps_ids_to_encoders() -> None:
    assert isinstance(build_encoder("stub", None, chunk=4, d_feature=8, n_tokens=4, image_size=16), StubEncoder)
    assert ENCODERS["siglip"] is FrozenSigLIP
    assert ENCODERS["pe_core"] is FrozenPerceptionEncoder
    assert ENCODERS["dinov3"] is FrozenDINOv3
    with pytest.raises(ValueError, match="unknown encoder"):
        build_encoder("resnet50", None, chunk=4, d_feature=8, n_tokens=4, image_size=16)


def test_channel_projection_commutes_with_the_filters() -> None:
    """Projecting to ``d_resid`` before filtering equals projecting afterwards."""
    from mars.models.mars.streams import ResidualStreams

    cfg = small().model
    streams = ResidualStreams(cfg, 8)
    features = torch.randn(2, 8, 16, cfg.d_feature)
    projection = torch.nn.Linear(cfg.d_feature, cfg.d_resid, bias=False)
    valid = prefix(8, 5)
    with torch.no_grad():
        before = streams(projection(features), valid)
        after = [projection(s) for s in streams(features, valid)]
    for a, b in zip(before, after):
        assert torch.allclose(a, b, atol=1e-4)


def test_detector_registers_under_its_name() -> None:
    raw = {
        "data": {"train": ["ffpp"], "test": ["ffpp"]},
        "model": {"name": "mars", "encoder_id": "stub"},
        "log": {"run_name": "mars_test"},
    }
    detector = build_detector("mars", raw)
    assert detector.name == "mars"
    assert detector.cfg.model.streams.n_residual == 7


def test_outputs_land_under_the_run_name() -> None:
    cfg = small("log.run_name=some_run")
    assert cfg.checkpoint_path.as_posix() == "checkpoints/mars/some_run"
    assert cfg.results_path.as_posix() == "results/mars/some_run"


def test_a_group_nests_the_run() -> None:
    """One paper table is one group, so its runs sit together on disk and in W&B."""
    cfg = small("log.group=streams_ablation", "log.run_name=no_spatial")
    assert cfg.checkpoint_path.as_posix() == "checkpoints/mars/streams_ablation/no_spatial"
    assert cfg.results_path.as_posix() == "results/mars/streams_ablation/no_spatial"
