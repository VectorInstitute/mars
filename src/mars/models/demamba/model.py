"""DeMamba network: X-CLIP B/16 with a Mamba over each spatial zone across time.

X-CLIP encodes every frame, exchanging messages within groups of consecutive
frames. Global branch: the pooled class tokens averaged over genuine frames. Detail
branch: the patch grid is split into zones; each zone's tokens are read in a snake
order within the frame, frame after frame, through one Mamba block, and averaged
over genuine frames. The two are concatenated and mapped to one logit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from mars.models.demamba.config import ModelConfig
from mars.models.demamba.mamba import ResidualMamba
from mars.models.demamba.xclip import XCLIPVisionTransformer, load_xclip_weights
from mars.utils.masking import masked_mean


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DeMambaOutput:
    """Forward-pass tensors: (B, 2) logits, real fixed at 0, fake the score."""

    logits: torch.Tensor


def snake_order(zone: int) -> torch.Tensor:
    """Column-wise snake over a zone x zone grid, as the released continuous scan."""
    order: list[int] = []
    for col in range(zone):
        rows = range(zone) if col % 2 == 0 else range(zone - 1, -1, -1)
        order.extend(row * zone + col for row in rows)
    return torch.tensor(order)


class DeMamba(nn.Module):
    """X-CLIP encoder, a global branch and a zone-wise Mamba detail branch."""

    mean: torch.Tensor
    std: torch.Tensor
    order: torch.Tensor

    def __init__(self, cfg: ModelConfig, image_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = XCLIPVisionTransformer(
            image_size, cfg.patch_size, cfg.width, cfg.layers, cfg.heads, cfg.hidden, cfg.message_group
        )
        if cfg.pretrained:
            load_xclip_weights(self.encoder, cfg.encoder_id)
        self.zones = (image_size // cfg.patch_size // cfg.zone) ** 2
        self.mamba = ResidualMamba(cfg.width, cfg.d_state, cfg.expand, cfg.d_conv, cfg.drop_path)
        self.fc_norm = nn.LayerNorm(self.zones * cfg.width)
        self.fc_norm2 = nn.LayerNorm(cfg.width)
        self.fc1 = nn.Linear((self.zones + 1) * cfg.width, 1)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("order", snake_order(cfg.zone), persistent=False)

    @property
    def classifier(self) -> nn.Linear:
        """The final linear layer."""
        return self.fc1

    def zone_sequences(self, patches: torch.Tensor) -> torch.Tensor:
        """Order (B, T, P, D) patches as zone sequences, snake then time."""
        batch, n_frames, n_patches, width = patches.shape
        z = self.cfg.zone
        side = round(n_patches**0.5) // z
        x = patches.reshape(batch, n_frames, side, z, side, z, width).permute(0, 2, 4, 1, 3, 5, 6)
        x = x.reshape(batch * side * side, n_frames, z * z, width)[:, :, self.order]
        return x.reshape(batch * side * side, n_frames * z * z, width)

    def forward(self, frames: torch.Tensor, valid: torch.Tensor) -> DeMambaOutput:
        """Classify (B, T, 3, S, S) frames; valid (B, T) marks genuine frames."""
        batch, n_frames = frames.shape[:2]
        valid = valid.to(device=frames.device, dtype=torch.bool)
        frames = (frames * 0.5 + 0.5 - self.mean) / self.std
        tokens, pooled = self.encoder(frames, valid)
        global_feature = self.fc_norm2(masked_mean(pooled, valid, dim=1))

        sequences = self.zone_sequences(tokens[:, :, 1:])
        # The scan is numerically sensitive, so it runs in float32.
        with torch.autocast(frames.device.type, enabled=False):
            mixed = checkpoint(self.mamba, sequences.float(), use_reentrant=False)
        steps = valid.repeat_interleave(self.cfg.zone**2, dim=1).repeat_interleave(self.zones, dim=0)
        detail = self.fc_norm(masked_mean(mixed, steps, dim=1).reshape(batch, -1))

        score = self.fc1(torch.cat([global_feature, detail.to(global_feature.dtype)], dim=1))
        return DeMambaOutput(logits=torch.cat([torch.zeros_like(score), score], dim=1))


def build_model(cfg: ModelConfig, image_size: int) -> DeMamba:
    """Construct DeMamba; load X-CLIP's released weights unless pretrained is false."""
    return DeMamba(cfg, image_size)
