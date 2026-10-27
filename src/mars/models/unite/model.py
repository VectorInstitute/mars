"""UNITE model assembly.

Frames go through a frozen encoder, a per-token temporal transformer, then a
spatial reduction and linear classifier. Block-1 head contributions form A and
are pooled against encoder features to give P for the AD loss. Kept as published:
every frame, padded or not, enters attention and the temporal mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mars.models.unite.config import ModelConfig
from mars.models.unite.encoder import build_encoder
from mars.models.unite.pooling import AttentionPool, build_attention, pooled_features
from mars.models.unite.positional import TemporalPositionalEncoding
from mars.models.unite.transformer import TemporalTransformer


@dataclass
class UniteOutput:
    """Forward-pass tensors."""

    logits: torch.Tensor
    pooled: torch.Tensor | None
    attention: torch.Tensor | None
    patch_weights: torch.Tensor | None = None


class UNITE(nn.Module):
    """Clip-level real/fake classifier."""

    def __init__(self, cfg: ModelConfig, encoder: nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        if self.encoder.d_feature != cfg.d_feature or self.encoder.n_tokens != cfg.n_tokens:
            raise ValueError(
                f"encoder emits ({self.encoder.n_tokens}, {self.encoder.d_feature}) "
                f"but config expects ({cfg.n_tokens}, {cfg.d_feature})"
            )
        self.input_proj: nn.Module = (
            nn.Identity() if cfg.d_model == cfg.d_feature else nn.Linear(cfg.d_feature, cfg.d_model)
        )
        self.positional = TemporalPositionalEncoding(cfg.d_model)
        self.input_norm: nn.Module = nn.LayerNorm(cfg.d_model) if cfg.input_norm else nn.Identity()
        self.transformer = TemporalTransformer(
            cfg.d_model,
            cfg.depth,
            cfg.n_heads,
            cfg.mlp_ratio,
            cfg.dropout,
            cfg.norm_style,
            cfg.grad_checkpoint,
        )
        self.pool = AttentionPool(cfg.d_model, cfg.pool_hidden) if cfg.spatial_pool == "attn" else None
        self.classifier = nn.Linear(cfg.d_model, cfg.n_classes)

    def forward(self, frames: torch.Tensor, valid: torch.Tensor | None = None) -> UniteOutput:
        """Classify (B, n_f, 3, S, S) frames.

        UNITE attends over every frame, so valid is accepted and unused. The
        pooled features and attention the AD loss needs are built only in training.
        """
        need_ad = self.training
        xi = self.encoder(frames).to(dtype=self.classifier.weight.dtype)
        batch, n_frames, n_tokens, _ = xi.shape

        tokens = self.input_norm(self.positional(self.input_proj(xi)))
        tokens = tokens.permute(0, 2, 1, 3).reshape(batch * n_tokens, n_frames, -1)

        hidden, heads = self.transformer(tokens, probe_block=0 if need_ad else -1)
        del tokens
        per_token = hidden.mean(dim=1).reshape(batch, n_tokens, -1)
        assert per_token.shape[:2] == (batch, n_tokens), per_token.shape

        if self.pool is None:
            pooled_hidden, patch_weights = per_token.mean(dim=1), None
        else:
            pooled_hidden, patch_weights = self.pool(per_token)
        logits = self.classifier(pooled_hidden)

        attention = pooled = None
        if need_ad:
            attention = build_attention(heads, batch, n_tokens)
            pooled = pooled_features(attention, xi)
        return UniteOutput(
            logits=logits,
            pooled=pooled,
            attention=attention,
            patch_weights=patch_weights,
        )


def build_model(cfg: ModelConfig, image_size: int = 384) -> UNITE:
    """Construct UNITE with the frozen encoder named by encoder_id."""
    encoder = build_encoder(
        cfg.encoder_id,
        cfg.encoder_path,
        chunk=cfg.encoder_chunk,
        d_feature=cfg.d_feature,
        n_tokens=cfg.n_tokens,
        image_size=image_size,
    )
    return UNITE(cfg, encoder)
