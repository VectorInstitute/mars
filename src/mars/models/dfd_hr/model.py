"""DFD-HR network: CLIP ViT-L/14 with MoE adapters and hierarchical routing, per frame.

Each genuine frame is seen twice: resized to one global view, and cut into 2 x 2
tiles at twice the resolution. Both pass through frozen CLIP with trainable MoE
adapters after every attention and MLP. In the routed blocks, a layer router lets
a view stop early and a token router sends only the top-scoring tokens through,
re-weighted by their scores. The global class token and a query-attention summary
of the four tile class tokens are classified together; the clip logit is the mean
of frame logits over genuine frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from mars.models.dfd_hr.clip import CLIPVisionTransformer, ResidualAttentionBlock, load_openai_weights
from mars.models.dfd_hr.config import ModelConfig
from mars.models.dfd_hr.routing import LayerRouter, MoEAdapter, TokenRouter, gumbel_sigmoid, spearman_corr
from mars.utils.masking import masked_mean


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class DFDHROutput:
    """Forward-pass tensors.

    Attributes
    ----------
    logits : torch.Tensor
        (B, C) clip logits, the mean over genuine frames.
    frame_logits : torch.Tensor
        (B, T, C) per-frame logits; zero at padded frames.
    valid : torch.Tensor
        (B, T) mask of genuine frames.
    ranking : torch.Tensor
        Scalar 1 minus the mean Spearman correlation between token scores and
        similarity to the global class token; rank based, so it has no gradient.
    """

    logits: torch.Tensor
    frame_logits: torch.Tensor
    valid: torch.Tensor
    ranking: torch.Tensor


def tiles(x: torch.Tensor, n: int) -> torch.Tensor:
    """Cut (N, C, H, W) into n x n tiles, stacked tile-major along the batch."""
    batch, channels, height, width = x.shape
    x = x.reshape(batch, channels, n, height // n, n, width // n)
    return x.permute(2, 4, 0, 1, 3, 5).reshape(n * n * batch, channels, height // n, width // n)


class DFDHR(nn.Module):
    """Frozen CLIP with MoE adapters, token and layer routing, and a two-view head."""

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

        width = cfg.clip_width
        self.adapters_attn = nn.ModuleList(
            MoEAdapter(width, cfg.n_experts, cfg.top_k, cfg.expert_ratio) for _ in range(cfg.clip_layers)
        )
        self.adapters_mlp = nn.ModuleList(
            MoEAdapter(width, cfg.n_experts, cfg.top_k, cfg.expert_ratio) for _ in range(cfg.clip_layers)
        )
        routed = [str(block) for block in cfg.routed_blocks]
        self.token_routers = nn.ModuleDict({block: TokenRouter(width) for block in routed})
        self.layer_routers = nn.ModuleDict({block: LayerRouter(width, cfg.router_hidden) for block in routed})
        # As released, the exit decision reaches the loss only through indexing, so
        # the layer routers never receive a gradient and keep their initial weights.
        self.layer_routers.requires_grad_(False)

        self.query_token = nn.Parameter(torch.randn(1, 1, cfg.clip_dim))
        self.query_attn = nn.MultiheadAttention(cfg.clip_dim, cfg.query_heads, batch_first=True)
        self.head = nn.Linear(2 * cfg.clip_dim, cfg.n_classes)
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    @property
    def classifier(self) -> nn.Linear:
        """The final linear layer."""
        return self.head

    def train(self, mode: bool = True) -> DFDHR:
        """Train everything but CLIP, which stays in eval."""
        super().train(mode)
        self.encoder.eval()
        return self

    def adapted(self, index: int, block: ResidualAttentionBlock, x: torch.Tensor) -> torch.Tensor:
        """One CLIP block with an MoE adapter after its attention and after its MLP."""
        x = x + self.adapters_attn[index](block.attend(block.ln_1(x), None))
        return x + self.adapters_mlp[index](block.mlp(block.ln_2(x)))

    def routed(self, index: int, block: ResidualAttentionBlock, x: torch.Tensor) -> torch.Tensor:
        """Run the adapted block on the top-scoring tokens, re-weighted by score."""
        scores = self.token_routers[str(index + 1)](x)
        k = max(1, int(self.cfg.capacity * x.shape[1]))
        chosen = scores.topk(k, dim=1).indices.sort(dim=1).values
        gather = chosen[..., None].expand(-1, -1, x.shape[-1])
        updated = self.adapted(index, block, x.gather(1, gather))
        weight = F.softmax(scores.gather(1, chosen), dim=1)[..., None]
        return x.scatter(1, gather, (updated * (1 + weight)).to(x.dtype))

    def encode(self, pixels: torch.Tensor, guide: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode CLIP-normalised (N, 3, S, S) views to the last class token, ranking.

        guide is the global view's final class token for each tile; the ranking
        term compares the first routed block's token scores with similarity to it.
        """
        x = self.encoder.embed(pixels)
        active = torch.ones(len(x), dtype=torch.bool, device=x.device)
        ranking = x.new_zeros((), dtype=torch.float32)
        for index, block in enumerate(self.encoder.blocks):
            if str(index + 1) not in self.layer_routers:
                x = checkpoint(self.adapted, index, block, x, use_reentrant=False)
                continue
            logits = self.layer_routers[str(index + 1)](x[:, 0])
            go = gumbel_sigmoid(logits) > 0.5 if self.training else torch.sigmoid(logits) > 0.5
            active = active & go
            if not active.any():
                continue
            rows = active.nonzero(as_tuple=True)[0]
            if guide is not None and index + 1 == self.cfg.routed_blocks[0]:
                scores = self.token_routers[str(index + 1)](x[rows])
                similarity = torch.einsum(
                    "nld,nd->nl", F.normalize(x[rows].float(), dim=-1), F.normalize(guide[rows].float(), dim=-1)
                )
                ranking = 1 - spearman_corr(scores, similarity).mean()
            x = x.index_put((rows,), checkpoint(self.routed, index, block, x[rows], use_reentrant=False))
        return x[:, 0], ranking

    def frame_logits(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, 3, 2S, 2S) frames in [-1, 1] to (N, C) logits and the ranking term."""
        pixels = (pixels * 0.5 + 0.5 - self.mean) / self.std
        size = self.cfg.clip_size
        global_view = F.interpolate(pixels.float(), size=(size, size), mode="bicubic").to(pixels.dtype)
        global_cls, _ = self.encode(global_view, None)
        tile_cls, ranking = self.encode(tiles(pixels, 2), global_cls.repeat(4, 1))

        def project(cls: torch.Tensor) -> torch.Tensor:
            return self.encoder.ln_post(cls) @ self.encoder.proj

        batch = len(pixels)
        tile_tokens = project(tile_cls).reshape(4, batch, -1).transpose(0, 1)
        summary, _ = self.query_attn(self.query_token.expand(batch, -1, -1), tile_tokens, tile_tokens)
        features = torch.cat([project(global_cls), summary[:, 0]], dim=-1)
        return self.head(features), ranking

    def forward(self, frames: torch.Tensor, valid: torch.Tensor) -> DFDHROutput:
        """Classify (B, T, 3, 2S, 2S) frames; valid (B, T) marks genuine frames."""
        batch, n_frames = frames.shape[:2]
        valid = valid.to(device=frames.device, dtype=torch.bool)
        genuine = frames.flatten(0, 1)[valid.flatten()]
        chunk = self.cfg.encoder_chunk
        parts = [self.frame_logits(genuine[i : i + chunk]) for i in range(0, len(genuine), chunk)]
        logits = torch.cat([part[0] for part in parts])
        frame_logits = logits.new_zeros(batch, n_frames, logits.shape[-1])
        frame_logits[valid] = logits
        return DFDHROutput(
            logits=masked_mean(frame_logits, valid, dim=1),
            frame_logits=frame_logits,
            valid=valid,
            ranking=torch.stack([part[1] for part in parts]).mean(),
        )


def build_model(cfg: ModelConfig) -> DFDHR:
    """Construct DFD-HR, loading CLIP's released weights unless pretrained is false."""
    return DFDHR(cfg)
