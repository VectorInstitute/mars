"""UNITE's frozen frame encoder: SigLIP, or a stub for CPU tests."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


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


class FrozenSigLIP(FrozenEncoder):
    """SigLIP vision tower (transformers); every output token is a patch."""

    def __init__(self, model_id: str, local_path: str | None, chunk: int, *, d_feature: int, n_tokens: int) -> None:
        from transformers import SiglipVisionModel

        super().__init__(SiglipVisionModel.from_pretrained(local_path or model_id), chunk, d_feature, n_tokens)

    def tokens(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return the last hidden state."""
        return self.backbone(pixel_values=pixels).last_hidden_state


class StubEncoder(nn.Module):
    """Fixed-init conv stand-in used by CPU tests. Same contract as FrozenEncoder."""

    def __init__(self, d_feature: int, n_tokens: int, image_size: int = 64) -> None:
        super().__init__()
        self.d_feature = d_feature
        self.n_tokens = n_tokens
        grid = int(round(n_tokens**0.5))
        self.patch = nn.Conv2d(3, d_feature, kernel_size=image_size // grid, stride=image_size // grid)
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            self.patch.weight.copy_(torch.randn(self.patch.weight.shape, generator=generator) * 0.02)
            if self.patch.bias is not None:
                self.patch.bias.zero_()
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> StubEncoder:  # noqa: ARG002
        """Stay frozen."""
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, S, S) -> (B, T, n_tokens, d_feature)."""
        batch, time = frames.shape[:2]
        tokens = self.patch(frames.flatten(0, 1)).flatten(2).transpose(1, 2)
        return tokens[:, : self.n_tokens].reshape(batch, time, self.n_tokens, self.d_feature)


def build_encoder(
    encoder_id: str, encoder_path: str | None, *, chunk: int, d_feature: int, n_tokens: int, image_size: int
) -> nn.Module:
    """Build SigLIP, or the stub when encoder_id is stub.

    Parameters
    ----------
    encoder_id : str
        stub, or a SigLIP model id.
    encoder_path : str or None
        Local weights, used instead of downloading encoder_id.
    chunk : int
        Frames encoded per forward call.
    d_feature, n_tokens : int
        Expected feature width and patch-token count.
    image_size : int
        Input side, used only by the stub.

    Returns
    -------
    nn.Module
        Maps (B, T, 3, S, S) frames to (B, T, n_tokens, d_feature) tokens.
    """
    if encoder_id == "stub":
        return StubEncoder(d_feature, n_tokens, image_size)
    if "siglip" in encoder_id.lower():
        return FrozenSigLIP(encoder_id, encoder_path, chunk, d_feature=d_feature, n_tokens=n_tokens)
    raise ValueError(f"UNITE expects stub or a SigLIP id, got {encoder_id!r}")
