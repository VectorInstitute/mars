"""ForAda network: CLIP ViT-L/14 steered by the Forensics Adapter, per frame.

Each genuine frame goes through frozen CLIP once for features, the adapter turns
those into attention biases, and CLIP's blocks run again with learnable query
tokens that attend to the patches through the biases. The queries become frame
logits; the clip logit is their mean over genuine frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from mars.models.forada.adapter import Adapter
from mars.models.forada.clip import CLIPVisionTransformer, load_openai_weights
from mars.models.forada.config import ModelConfig
from mars.utils.masking import masked_mean


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class ForAdaOutput:
    """Forward-pass tensors: (B, C) clip logits, (B, T, C) frame logits, (B, T) mask."""

    logits: torch.Tensor
    frame_logits: torch.Tensor
    valid: torch.Tensor


class IntraBlock(nn.Module):
    """Bottleneck over CLIP patch tokens, added back at 0.05 inside the guided pass."""

    def __init__(self, width: int, hidden: int = 192) -> None:
        super().__init__()
        self.conv_first = nn.Conv1d(width, hidden, kernel_size=1)
        self.conv_second = nn.Conv1d(hidden, width, kernel_size=1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """(N, L, D) patches to a (N, L, D) residual."""
        x = F.relu(self.conv_first(patches.transpose(1, 2).float()))
        return self.conv_second(x).transpose(1, 2)


class QueryHead(nn.Module):
    """(N, Q, D) query tokens to (N, C) logits: collapse the queries, then an MLP."""

    def __init__(self, n_queries: int, dim: int, n_classes: int) -> None:
        super().__init__()
        q = n_queries
        self.reduce = nn.Sequential(
            nn.Conv1d(q, q // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(q // 2),
            nn.ReLU(),
            nn.Conv1d(q // 2, q // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(q // 4),
            nn.ReLU(),
            nn.Conv1d(q // 4, 1, kernel_size=3, padding=1),
        )
        self.mlp = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim // 4), nn.ReLU())
        self.classifier = nn.Linear(dim // 4, n_classes)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """Collapse queries to one vector per frame and classify it."""
        return self.classifier(self.mlp(self.reduce(queries).squeeze(1)))


def query_mask(bias: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Additive attention mask over [queries, class token, patches], (N, heads, T, T).

    Queries see only themselves and the patches, the latter through bias; the class
    token and the patches never see the queries.
    """
    batch, heads, q = bias.shape[:3]
    size = q + n_tokens
    mask = bias.new_zeros(batch, heads, size, size)
    mask[:, :, :, :q] = -100
    mask[:, :, torch.arange(q), torch.arange(q)] = 0
    mask[:, :, :q, q] = -100
    mask[:, :, :q, q + 1 :] = bias.flatten(3)
    return mask


class ForAda(nn.Module):
    """Frozen CLIP, the Forensics Adapter, a guided CLIP pass and a query head."""

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
        self.adapter = Adapter(
            cfg.adapter_id,
            cfg.adapter_blocks,
            cfg.adapter_patch,
            cfg.n_queries,
            cfg.fusion_map,
            cfg.clip_width,
            cfg.clip_heads,
            cfg.mlp_dim,
            cfg.mlp_out_dim,
        )
        self.intra = IntraBlock(cfg.clip_width)
        self.head = QueryHead(cfg.n_queries, cfg.clip_dim, cfg.n_classes)
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    @property
    def classifier(self) -> nn.Linear:
        """The final linear layer."""
        return self.head.classifier

    def train(self, mode: bool = True) -> ForAda:
        """Train everything but CLIP, which stays in eval."""
        super().train(mode)
        self.encoder.eval()
        return self

    def guided(self, tokens: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Rerun CLIP's blocks with query tokens; return unit (N, Q, D) queries."""
        q = self.cfg.n_queries
        start = q + 1
        mask = query_mask(bias, tokens.shape[1])
        x = torch.cat([tokens[:, :1].expand(-1, q, -1), tokens], dim=1)
        for index, block in enumerate(self.encoder.blocks, start=1):
            x = checkpoint(block, x, mask, use_reentrant=False)
            if index == self.cfg.intra_block:
                patches = x[:, start:]
                x = torch.cat([x[:, :start], patches + 0.05 * self.intra(patches).to(patches.dtype)], dim=1)
        queries = self.encoder.ln_post(x[:, :q]) @ self.encoder.proj
        return F.normalize(queries, dim=-1)

    def frame_logits(self, pixels: torch.Tensor) -> torch.Tensor:
        """(N, 3, S, S) frames in [-1, 1] to (N, C) logits."""
        pixels = (pixels * 0.5 + 0.5 - self.mean) / self.std
        clip_pixels = F.interpolate(pixels, size=(self.cfg.clip_size,) * 2, mode="bilinear", align_corners=False)
        with torch.no_grad():
            layers = self.encoder.features(clip_pixels, {0, *self.cfg.fusion_map.values()})
        bias = self.adapter(pixels, {k: v[:, 1:] for k, v in layers.items()})
        return self.head(self.guided(layers[0], bias).float())

    def forward(self, frames: torch.Tensor, valid: torch.Tensor) -> ForAdaOutput:
        """Classify (B, T, 3, S, S) frames; valid (B, T) marks genuine frames."""
        batch, n_frames = frames.shape[:2]
        valid = valid.to(device=frames.device, dtype=torch.bool)
        genuine = frames.flatten(0, 1)[valid.flatten()]
        chunk = self.cfg.encoder_chunk
        logits = torch.cat([self.frame_logits(genuine[i : i + chunk]) for i in range(0, len(genuine), chunk)])
        frame_logits = logits.new_zeros(batch, n_frames, logits.shape[-1])
        frame_logits[valid] = logits
        return ForAdaOutput(logits=masked_mean(frame_logits, valid, dim=1), frame_logits=frame_logits, valid=valid)


def build_model(cfg: ModelConfig) -> ForAda:
    """Construct ForAda, loading CLIP's released weights unless pretrained is false."""
    return ForAda(cfg)
