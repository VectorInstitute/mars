"""GenD's Perception Encoder, frozen except where the model unfreezes its norms."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class FrozenEncoder(nn.Module):
    """Encode each frame independently to patch tokens with frozen weights.

    Subclasses load the backbone and implement tokens. d_feature and
    n_tokens come from the model YAML and are checked against the output.

    Parameters
    ----------
    backbone : nn.Module
        Pretrained vision model; frozen and kept in eval mode.
    chunk : int
        Frames encoded per forward call.
    d_feature, n_tokens : int
        Expected feature width and patch-token count.
    """

    def __init__(self, backbone: nn.Module, chunk: int, d_feature: int, n_tokens: int) -> None:
        super().__init__()
        backbone.eval().requires_grad_(False)
        self.backbone: Any = backbone
        self.chunk = chunk
        self.d_feature = d_feature
        self.n_tokens = n_tokens

    def train(self, mode: bool = True) -> FrozenEncoder:
        """Stay frozen: the backbone remains in eval."""
        super().train(mode)
        self.backbone.eval()
        return self

    def tokens(self, pixels: torch.Tensor) -> torch.Tensor:
        """Map (N, 3, S, S) pixels in [-1, 1] to patch tokens."""
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, S, S) -> (B, T, n_tokens, d_feature)."""
        batch, time = frames.shape[:2]
        flat = frames.flatten(0, 1)
        tokens = torch.cat([self.tokens(flat[i : i + self.chunk]) for i in range(0, flat.shape[0], self.chunk)])
        if tokens.shape[1:] != (self.n_tokens, self.d_feature):
            raise ValueError(
                f"encoder emits {tuple(tokens.shape[1:])}, config expects {(self.n_tokens, self.d_feature)}"
            )
        return tokens.reshape(batch, time, self.n_tokens, self.d_feature)


class FrozenPerceptionEncoder(FrozenEncoder):
    """Meta Perception Encoder (timm); prefix tokens are dropped.

    With layer set, tokens are read after that block, before the final norm.
    """

    def __init__(
        self,
        model_id: str,
        local_path: str | None,
        chunk: int,
        *,
        d_feature: int,
        n_tokens: int,
        layer: int | None = None,
    ) -> None:
        import timm

        overlay = {"file": local_path} if local_path else None
        backbone = timm.create_model(model_id, pretrained=True, pretrained_cfg_overlay=overlay)
        super().__init__(backbone, chunk, d_feature, n_tokens)
        self.layer = layer
        self.depth = len(backbone.blocks)
        self.n_prefix = backbone.num_prefix_tokens

    def hidden(self, pixels: torch.Tensor, depth: int) -> tuple[torch.Tensor, Any]:
        """Run the first depth blocks; return every token and the rotary embedding."""
        # Mirrors timm Eva.forward_features, stopping after block depth.
        x, rope = self.backbone._pos_embed(self.backbone.patch_embed(pixels))
        x = self.backbone.norm_pre(x)
        for block in self.backbone.blocks[:depth]:
            x = block(x, rope=rope)
        return x, rope

    def finish(self, x: torch.Tensor, rope: Any, start: int) -> torch.Tensor:
        """Run blocks from start, the final norm and the attention pool."""
        for block in self.backbone.blocks[start:]:
            x = checkpoint(block, x, rope=rope, use_reentrant=False)
        return self.backbone.forward_head(self.backbone.norm(x), pre_logits=True)

    def tune_layernorms(self, start: int) -> None:
        """Train LayerNorms from block start on, plus the final norm and the pool."""
        for name, module in self.backbone.named_modules():
            if not isinstance(module, nn.LayerNorm):
                continue
            head, _, rest = name.partition(".")
            late = int(rest.split(".")[0]) >= start if head == "blocks" else head not in ("patch_embed", "norm_pre")
            if late:
                module.requires_grad_(True)

    def tokens(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return patch tokens: final normed tokens, or those after block layer."""
        if self.layer is None:
            return self.backbone.forward_features(pixels)[:, self.n_prefix :]
        return self.hidden(pixels, self.layer)[0][:, self.n_prefix :]
