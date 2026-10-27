"""Unite network, detector factory, and losses."""

from __future__ import annotations

import torch

from mars.config import parse_overrides
from mars.losses.ce import cross_entropy_loss
from mars.models.factory import build_detector
from mars.models.unite.config import ADLossConfig, Config, load_config
from mars.models.unite.model import build_model
from mars.models.unite.objective import AttentionDiversityLoss


def test_ce_and_ad_objective_cpu() -> None:
    logits = torch.tensor([[2.0, 0.1], [0.2, 1.5]])
    labels = torch.tensor([0, 1])
    ce = cross_entropy_loss(logits, labels)
    assert ce.ndim == 0 and ce.item() > 0

    ad = AttentionDiversityLoss(2, 3, 4, ADLossConfig(delta_between=100.0, feature_norm="none"))
    pooled = torch.randn(4, 3, 4, requires_grad=True)
    ad(pooled, torch.tensor([0, 1, 0, 1])).backward()
    assert pooled.grad is not None
    assert pooled.grad.abs().sum() > 0


def stub_config() -> Config:
    """UNITE at stub-encoder size."""
    return load_config(
        "configs/models/unite/config.yaml",
        parse_overrides(
            [
                "model.encoder_id=stub",
                "model.d_feature=32",
                "model.n_tokens=16",
                "model.d_model=32",
                "model.depth=2",
                "model.n_heads=2",
                "data.n_frames=8",
                "data.image_size=64",
            ]
        ),
    )


def test_stub_encoder_forward_shapes() -> None:
    cfg = stub_config()
    model = build_model(cfg.model, cfg.data.image_size)
    frames = torch.randn(2, cfg.data.n_frames, 3, cfg.data.image_size, cfg.data.image_size)
    out = model.train()(frames, valid=torch.ones(2, cfg.data.n_frames, dtype=torch.bool))
    assert out.logits.shape == (2, 2)
    assert out.pooled is not None and out.pooled.shape == (2, cfg.model.n_heads, cfg.data.n_frames)
    detector = build_detector(
        "unite",
        {
            "model": {"name": "unite"},
            "data": {"train": "ffpp", "test": ["ffpp"]},
            "log": {"run_name": "stub"},
        },
    )
    assert detector.name == "unite"
