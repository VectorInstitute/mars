"""The Forensics Adapter: a small ViT that reads CLIP and emits attention biases.

Learnable queries and the frame's patches run through the first blocks of a ViT-Tiny;
CLIP patch tokens are added in at fixed blocks. The queries then decode, per CLIP
head, a bias over CLIP's patch grid.
"""

from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import nn


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channels of a (N, C, H, W) map."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise each position across channels."""
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MLP(nn.Module):
    """Layers with ReLU between them; affine is Linear or a 1x1 convolution."""

    def __init__(self, dims: list[int], affine: type[nn.Module] = nn.Linear) -> None:
        super().__init__()
        extra = {"kernel_size": 1} if affine is nn.Conv2d else {}
        self.layers = nn.ModuleList(affine(a, b, **extra) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply every layer, ReLU between them."""
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = F.relu(x)
        return x


class Fusion(nn.Module):
    """Add a projection of CLIP patch tokens to the adapter's patch tokens."""

    def __init__(self, clip_width: int, width: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(ChannelLayerNorm(clip_width), nn.Conv2d(clip_width, width, kernel_size=1))

    def forward(self, x: torch.Tensor, clip_tokens: torch.Tensor, side: int) -> torch.Tensor:
        """Adapter patches (N, L, width) plus 0.01 times projected CLIP tokens."""
        grid = clip_tokens.transpose(1, 2).reshape(clip_tokens.shape[0], -1, side, side)
        return x + 0.01 * self.proj(grid).flatten(2).transpose(1, 2)


class BiasDecoder(nn.Module):
    """Queries and patch features to one attention bias per CLIP head."""

    def __init__(self, width: int, hidden: int, out: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.query_mlp = MLP([width, hidden, hidden, out])
        self.attn_mlp = MLP([width, hidden, hidden, out * heads], affine=nn.Conv2d)
        self.bias_scaling = nn.Linear(1, 1)

    def forward(self, query: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """(N, Q, D) queries and (N, D, h, w) patches to a (N, heads, Q, h, w) bias."""
        query = self.query_mlp(query)
        attn = self.attn_mlp(patches)
        batch, _, h, w = attn.shape
        attn = attn.reshape(batch, self.heads, -1, h, w)
        bias = torch.einsum("nqd,nkdhw->nkqhw", query, attn)
        return self.bias_scaling(bias[..., None]).squeeze(-1)


class Adapter(nn.Module):
    """ViT-Tiny blocks over learnable queries and frame patches, fused with CLIP."""

    def __init__(
        self,
        vit_id: str,
        n_blocks: int,
        patch_size: int,
        n_queries: int,
        fusion_map: dict[int, int],
        clip_width: int,
        clip_heads: int,
        hidden: int,
        out: int,
    ) -> None:
        super().__init__()
        vit = timm.create_model(vit_id, pretrained=False, num_classes=0)
        width = vit.num_features
        self.n_queries = n_queries
        self.fusion_map = fusion_map
        self.blocks = vit.blocks[:n_blocks]
        self.pos_embed = nn.Parameter(vit.pos_embed[:, vit.num_prefix_tokens :].detach().clone())
        self.patch_conv = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size)
        self.query_embed = nn.Parameter(torch.randn(1, n_queries, width) * 0.02)
        self.query_pos_embed = nn.Parameter(torch.randn(1, n_queries, width) * 0.02)
        self.ln_pre = nn.LayerNorm(width)
        self.fusions = nn.ModuleDict({str(block): Fusion(clip_width, width) for block in fusion_map})
        self.decoder = BiasDecoder(width, hidden, out, clip_heads)

    def positions(self, side: int) -> torch.Tensor:
        """Query positions followed by patch positions resized to a side x side grid."""
        stored = self.pos_embed.shape[1]
        grid = round(stored**0.5)
        patch = self.pos_embed.transpose(1, 2).reshape(1, -1, grid, grid)
        patch = F.interpolate(patch, size=(side, side), mode="bilinear", align_corners=False)
        return torch.cat([self.query_pos_embed, patch.flatten(2).transpose(1, 2)], dim=1)

    def forward(self, pixels: torch.Tensor, clip_patches: dict[int, torch.Tensor]) -> torch.Tensor:
        """Frames and CLIP patch tokens by layer to a (N, heads, Q, h, w) bias."""
        x = self.patch_conv(pixels)
        side = x.shape[-1]
        x = x.flatten(2).transpose(1, 2)
        positions = self.positions(side)
        x = self.ln_pre(torch.cat([self.query_embed.expand(x.shape[0], -1, -1), x], dim=1) + positions)
        for index, block in enumerate(self.blocks, start=1):
            x = block(x)
            if index in self.fusion_map:
                fused = self.fusions[str(index)](x[:, self.n_queries :], clip_patches[self.fusion_map[index]], side)
                x = torch.cat([x[:, : self.n_queries], fused], dim=1)
            # Positions are added back after every block but the last.
            if index < len(self.blocks):
                x = x + positions
        patches = x[:, self.n_queries :].transpose(1, 2).reshape(x.shape[0], -1, side, side)
        return self.decoder(x[:, : self.n_queries], patches)
