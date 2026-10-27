"""GenD network: a Perception Encoder whose normalisation layers are tuned.

Each genuine frame is encoded on its own; the pooled feature is L2-normalised and
classified by one linear layer. The clip logit is the mean of the frame logits over
genuine frames, so padded frames are never encoded.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from mars.models.gend.config import ModelConfig
from mars.models.gend.encoder import FrozenPerceptionEncoder
from mars.utils.masking import masked_mean


@dataclass
class GenDOutput:
    """Forward-pass tensors.

    Attributes
    ----------
    logits : torch.Tensor
        (B, C) clip logits, the mean over genuine frames.
    frame_logits : torch.Tensor
        (B, T, C) per-frame logits; zero at padded frames.
    embeddings : torch.Tensor
        (B, T, D) unit-norm frame embeddings; zero at padded frames.
    valid : torch.Tensor
        (B, T) mask of genuine frames.
    """

    logits: torch.Tensor
    frame_logits: torch.Tensor
    embeddings: torch.Tensor
    valid: torch.Tensor


class GenD(nn.Module):
    """Perception Encoder with every norm trainable and an L2 head per frame."""

    def __init__(self, cfg: ModelConfig, image_size: int) -> None:
        super().__init__()
        encoder = FrozenPerceptionEncoder(
            cfg.encoder_id, cfg.encoder_path, cfg.encoder_chunk, d_feature=cfg.d_feature, n_tokens=cfg.n_tokens
        )
        # As in GenD: every encoder parameter whose name contains norm is trained.
        for name, parameter in encoder.backbone.named_parameters():
            if "norm" in name:
                parameter.requires_grad_(True)
        self.encoder = encoder
        self.classifier = nn.Linear(cfg.d_feature, cfg.n_classes)

    def embed(self, pixels: torch.Tensor) -> torch.Tensor:
        """Pooled encoder feature of (N, 3, S, S) frames, (N, D)."""
        tokens, rope = self.encoder.hidden(pixels, 0)
        return self.encoder.finish(tokens, rope, 0)

    def forward(self, frames: torch.Tensor, valid: torch.Tensor) -> GenDOutput:
        """Classify (B, T, 3, S, S) frames; valid (B, T) marks genuine frames."""
        batch, n_frames = frames.shape[:2]
        valid = valid.to(device=frames.device, dtype=torch.bool)
        genuine = frames.flatten(0, 1)[valid.flatten()]
        chunk = self.encoder.chunk
        pooled = torch.cat([self.embed(genuine[i : i + chunk]) for i in range(0, len(genuine), chunk)])

        unit = F.normalize(pooled.float(), dim=-1)
        logits = self.classifier(unit)
        frame_logits = logits.new_zeros(batch, n_frames, logits.shape[-1])
        frame_logits[valid] = logits
        embeddings = unit.new_zeros(batch, n_frames, unit.shape[-1])
        embeddings[valid] = unit
        return GenDOutput(
            logits=masked_mean(frame_logits, valid, dim=1),
            frame_logits=frame_logits,
            embeddings=embeddings,
            valid=valid,
        )


def build_model(cfg: ModelConfig, image_size: int) -> GenD:
    """Construct GenD."""
    return GenD(cfg, image_size)
