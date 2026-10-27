"""Effort network: CLIP ViT-L/14 whose attention projections adapt in an SVD residual.

Every query, key, value and output projection of every block is split into its
top singular directions, frozen, and a trainable rank-k residual. Each genuine
frame's class token, after CLIP's final norm, is classified by one linear layer;
the clip logit is the mean of frame logits over genuine frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from mars.models.effort.clip import CLIPVisionTransformer, ResidualAttentionBlock, load_openai_weights
from mars.models.effort.config import ModelConfig
from mars.models.effort.svd import SVDResidualLinear
from mars.utils.masking import masked_mean


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
PROJECTIONS = ("q", "k", "v", "out")


@dataclass
class EffortOutput:
    """Forward-pass tensors.

    Attributes
    ----------
    logits : torch.Tensor
        (B, C) clip logits, the mean over genuine frames.
    frame_logits : torch.Tensor
        (B, T, C) per-frame logits; zero at padded frames.
    valid : torch.Tensor
        (B, T) mask of genuine frames.
    regularisation : torch.Tensor
        Mean over projections of the orthogonality plus singular-value constraint;
        zero outside training.
    """

    logits: torch.Tensor
    frame_logits: torch.Tensor
    valid: torch.Tensor
    regularisation: torch.Tensor


def split_projections(block: ResidualAttentionBlock, residual_rank: int) -> nn.ModuleDict:
    """SVD-split query, key, value and output projections of one CLIP block."""
    weights = block.attn.in_proj_weight.chunk(3)
    biases = block.attn.in_proj_bias.chunk(3)
    layers = {name: SVDResidualLinear(w, b, residual_rank) for name, w, b in zip(PROJECTIONS, weights, biases)}
    layers["out"] = SVDResidualLinear(block.attn.out_proj.weight, block.attn.out_proj.bias, residual_rank)
    return nn.ModuleDict(layers)


class Effort(nn.Module):
    """Frozen CLIP with SVD-residual attention projections and a linear head."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = CLIPVisionTransformer(
            cfg.clip_size, cfg.clip_patch, cfg.clip_width, cfg.clip_layers, cfg.clip_heads, cfg.clip_dim
        )
        if cfg.pretrained:
            load_openai_weights(self.encoder, cfg.encoder_path)
        self.encoder.eval().requires_grad_(False)
        blocks = [cast(ResidualAttentionBlock, block) for block in self.encoder.blocks]
        self.projections = nn.ModuleList(split_projections(block, cfg.residual_rank) for block in blocks)
        self.head = nn.Linear(cfg.clip_width, cfg.n_classes)
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    @property
    def classifier(self) -> nn.Linear:
        """The final linear layer."""
        return self.head

    def train(self, mode: bool = True) -> Effort:
        """Train the residuals and head; CLIP stays in eval."""
        super().train(mode)
        self.encoder.eval()
        return self

    def block(self, block: ResidualAttentionBlock, layers: nn.ModuleDict, x: torch.Tensor) -> torch.Tensor:
        """Run one CLIP block with its split attention projections."""
        h = block.ln_1(x)
        batch, length, width = h.shape

        def heads(name: str) -> torch.Tensor:
            return layers[name](h).reshape(batch, length, self.cfg.clip_heads, -1).transpose(1, 2)

        attended = F.scaled_dot_product_attention(heads("q"), heads("k"), heads("v"))
        x = x + layers["out"](attended.transpose(1, 2).reshape(batch, length, width))
        return x + block.mlp(block.ln_2(x))

    def frame_logits(self, pixels: torch.Tensor) -> torch.Tensor:
        """(N, 3, S, S) frames in [-1, 1] to (N, C) logits."""
        x = self.encoder.embed((pixels * 0.5 + 0.5 - self.mean) / self.std)
        for block, layers in zip(self.encoder.blocks, self.projections):
            x = checkpoint(self.block, block, layers, x, use_reentrant=False)
        return self.head(self.encoder.ln_post(x[:, 0]))

    def regularisation(self) -> torch.Tensor:
        """Mean over split layers of the orthogonality and singular-value terms."""
        layers = [m for m in self.projections.modules() if isinstance(m, SVDResidualLinear)]
        with torch.autocast(self.head.weight.device.type, enabled=False):
            terms = [layer.orthogonal_loss() + layer.singular_value_loss() for layer in layers]
        return torch.stack(terms).mean()

    def forward(self, frames: torch.Tensor, valid: torch.Tensor) -> EffortOutput:
        """Classify (B, T, 3, S, S) frames; valid (B, T) marks genuine frames."""
        batch, n_frames = frames.shape[:2]
        valid = valid.to(device=frames.device, dtype=torch.bool)
        genuine = frames.flatten(0, 1)[valid.flatten()]
        chunk = self.cfg.encoder_chunk
        logits = torch.cat([self.frame_logits(genuine[i : i + chunk]) for i in range(0, len(genuine), chunk)])
        frame_logits = logits.new_zeros(batch, n_frames, logits.shape[-1])
        frame_logits[valid] = logits
        return EffortOutput(
            logits=masked_mean(frame_logits, valid, dim=1),
            frame_logits=frame_logits,
            valid=valid,
            regularisation=self.regularisation() if self.training else frames.new_zeros(()),
        )


def build_model(cfg: ModelConfig) -> Effort:
    """Construct Effort, loading CLIP's released weights unless pretrained is false."""
    return Effort(cfg)
