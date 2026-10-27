"""Per-spatial-token temporal transformer.

Each patch position is its own temporal sequence, so attention runs over time
only. Blocks are post-LayerNorm by default. An optional (B, T) mask of genuine
frames keeps padded frames out of every attention key; without it, or with no
padding, the output is unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from mars.utils.masking import masked_mean


class EncoderBlock(nn.Module):
    """One self-attention + MLP block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
        norm_style: str = "post",
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm_style = norm_style
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def head_contributions(self, attended: torch.Tensor, key_valid: torch.Tensor | None) -> torch.Tensor:
        """Per-head contribution to the block output, averaged over genuine frames.

        The temporal mean is taken before projecting through the output weight;
        the projection is linear, so the result matches projecting first.
        """
        mean = attended.mean(dim=2) if key_valid is None else masked_mean(attended, key_valid[:, None, :], dim=2)
        per_head = self.out.weight.view(-1, self.n_heads, self.head_dim)
        return torch.einsum("bhd,ohd->bho", mean, per_head)

    def attend(
        self, x: torch.Tensor, probe: bool, key_valid: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Multi-head attention over genuine keys; with probe, also head contributions.

        The mask enters before the softmax, so a padded frame cannot reach any
        query through the normalisation either.
        """
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn_mask = None if key_valid is None else key_valid[:, None, None, :].to(dtype=torch.bool)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        heads = self.head_contributions(attended, key_valid) if probe else None
        out = self.out(attended.transpose(1, 2).reshape(batch, length, -1))
        return out, heads

    def forward(
        self, x: torch.Tensor, probe: bool = False, key_valid: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply attention and MLP with residual connections."""
        if self.norm_style == "pre":
            attended, heads = self.attend(self.norm_attn(x), probe, key_valid)
            x = x + self.dropout(attended)
            return x + self.mlp(self.norm_mlp(x)), heads
        attended, heads = self.attend(x, probe, key_valid)
        x = self.norm_attn(x + self.dropout(attended))
        return self.norm_mlp(x + self.mlp(x)), heads


class TemporalTransformer(nn.Module):
    """Encoder stack over ``(B * t_s, n_f, d_model)`` per-position sequences."""

    def __init__(
        self,
        d_model: int,
        depth: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
        norm_style: str = "post",
        grad_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, n_heads, mlp_ratio, dropout, norm_style) for _ in range(depth)
        )

    def forward(
        self, x: torch.Tensor, probe_block: int = 0, key_valid: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (hidden, heads) from probe_block; heads is None if probe_block < 0."""
        probe_heads = None
        for index, block in enumerate(self.blocks):
            probe = index == probe_block
            if self.grad_checkpoint and self.training:
                x, heads = checkpoint(block, x, probe, key_valid, use_reentrant=False)
            else:
                x, heads = block(x, probe, key_valid)
            if probe:
                probe_heads = heads
        return x, probe_heads
