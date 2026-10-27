"""Per-spatial-token temporal transformer.

Each patch position is its own temporal sequence, so attention runs over time
only. Blocks are post-LayerNorm by default.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


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

    def head_contributions(self, attended: torch.Tensor) -> torch.Tensor:
        """Per-head contribution to the block output, averaged over time.

        The temporal mean is taken before projecting through ``W_O``. The
        projection is linear so the result matches projecting first.
        """
        per_head = self.out.weight.view(-1, self.n_heads, self.head_dim)
        return torch.einsum("bhd,ohd->bho", attended.mean(dim=2), per_head)

    def attend(self, x: torch.Tensor, probe: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Multi-head attention. When ``probe``, also return per-head contributions."""
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v)
        heads = self.head_contributions(attended) if probe else None
        out = self.out(attended.transpose(1, 2).reshape(batch, length, -1))
        return out, heads

    def forward(self, x: torch.Tensor, probe: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply attention and MLP with residual connections."""
        if self.norm_style == "pre":
            attended, heads = self.attend(self.norm_attn(x), probe)
            x = x + self.dropout(attended)
            return x + self.mlp(self.norm_mlp(x)), heads
        attended, heads = self.attend(x, probe)
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

    def forward(self, x: torch.Tensor, probe_block: int = 0) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(hidden, heads)`` from ``probe_block``.

        ``heads`` is ``None`` when ``probe_block < 0``.
        """
        probe_heads = None
        for index, block in enumerate(self.blocks):
            probe = index == probe_block
            if self.grad_checkpoint and self.training:
                x, heads = checkpoint(block, x, probe, use_reentrant=False)
            else:
                x, heads = block(x, probe)
            if probe:
                probe_heads = heads
        return x, probe_heads
