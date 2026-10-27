"""MARS residual path: multiscale appearance-residual streams over frozen tokens.

Frames are encoded once and frozen. Each patch token is projected to an
appearance stream and to a set of fixed band-pass residual streams over the patch
grid and the frame axis; the streams are summed, stamped positionally, and read
by a temporal transformer. The temporal reduction runs before the spatial pool,
so every patch is summarised over time and the pool then weighs patches.

Padded observations are excluded at every temporal site: the residual filters,
the positional stamp, the attention keys, and the temporal mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mars.models.mars.config import ModelConfig
from mars.models.mars.encoder import build_encoder
from mars.models.mars.pooling import AttentionPool
from mars.models.mars.positional import TemporalPositionalEncoding
from mars.models.mars.streams import StreamFusion
from mars.models.mars.transformer import TemporalTransformer
from mars.models.mars.validity import is_prefix, observation_index
from mars.utils.masking import masked_mean


@dataclass
class MarsOutput:
    """Forward-pass tensors."""

    logits: torch.Tensor
    pooled: torch.Tensor | None = None
    patch_weights: torch.Tensor | None = None
    path_logits: tuple[torch.Tensor, ...] = ()


class ResidualPath(nn.Module):
    """Frozen encoder, multiscale stream fusion, temporal trunk, spatial pool.

    Runs alone as single-path MARS, or inside Mars on tokens it already has.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        n_frames: int,
        image_size: int = 384,
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        if encoder is None:
            encoder = build_encoder(
                cfg.encoder_id,
                cfg.encoder_path,
                chunk=cfg.encoder_chunk,
                d_feature=cfg.d_feature,
                n_tokens=cfg.n_tokens,
                image_size=image_size,
                layer=cfg.encoder_layer,
            )
        self.encoder = encoder
        if self.encoder.d_feature != cfg.d_feature or self.encoder.n_tokens != cfg.n_tokens:
            raise ValueError(
                f"encoder emits {self.encoder.d_feature}x{self.encoder.n_tokens} tokens, "
                f"config expects {cfg.d_feature}x{cfg.n_tokens}"
            )
        self.fusion = StreamFusion(cfg, n_frames)
        self.positional = TemporalPositionalEncoding(cfg.d_model)
        self.block_norm: nn.Module = nn.LayerNorm(cfg.d_model) if cfg.block_norm else nn.Identity()
        trunk = (cfg.d_model, cfg.depth, cfg.n_heads, cfg.mlp_ratio, cfg.dropout, cfg.norm_style, cfg.grad_checkpoint)
        if cfg.temporal_validity:
            # The locked MARS drew its masked trunk after a discarded plain one; drawing
            # it twice keeps every weight from here on at its locked initial value.
            TemporalTransformer(*trunk)
        self.transformer = TemporalTransformer(*trunk)
        self.pool = AttentionPool(cfg.d_model, cfg.pool_hidden) if cfg.spatial_pool == "attn" else None
        self.classifier = nn.Linear(cfg.d_model, cfg.n_classes)

    def resolve_validity(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor | None:
        """Return the (B, T) mask of genuine frames, or None when masking is off."""
        if not self.cfg.temporal_validity:
            return None
        valid = valid.to(device=features.device, dtype=torch.bool)
        if not is_prefix(valid):
            raise ValueError("validity mask must be a contiguous prefix of genuine observations")
        return valid

    def trunk(
        self,
        frames: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[int, int, int]]:
        """Per-frame, per-token hidden states from the temporal transformer.

        Returns the hidden states as (B*N, T, d_model), the key mask aligned
        with them, and the (batch, n_frames, n_tokens) shape triple.
        """
        features = self.encoder(frames).to(dtype=self.classifier.weight.dtype)
        batch, n_frames, n_tokens, _ = features.shape
        mask = self.resolve_validity(features, valid)

        fused = self.fusion(features, mask)
        stamped = self.positional(fused, observation_index(mask) if mask is not None else None)
        tokens = self.block_norm(stamped).permute(0, 2, 1, 3).reshape(batch * n_tokens, n_frames, -1)

        key_valid = None
        if mask is not None:
            key_valid = mask[:, None, :].expand(batch, n_tokens, n_frames).reshape(batch * n_tokens, n_frames)
        hidden, _ = self.transformer(tokens, -1, key_valid)
        return hidden, key_valid, (batch, n_frames, n_tokens)

    def forward(self, frames: torch.Tensor, valid: torch.Tensor) -> MarsOutput:
        """Classify (B, T, 3, S, S) frames; valid (B, T) marks genuine frames."""
        hidden, key_valid, (batch, _, n_tokens) = self.trunk(frames, valid)

        pooled_time = hidden.mean(dim=1) if key_valid is None else masked_mean(hidden, key_valid, dim=1)
        per_token = pooled_time.reshape(batch, n_tokens, -1)

        if self.pool is None:
            pooled, patch_weights = per_token.mean(dim=1), None
        else:
            pooled, patch_weights = self.pool(per_token)
        return MarsOutput(logits=self.classifier(pooled), pooled=pooled, patch_weights=patch_weights)
