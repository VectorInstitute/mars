"""Frozen frame encoders: SigLIP, Perception Encoder and DINOv3."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from mars.models.mars.lora import LoRALinear


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


def tune_late_layernorms(backbone: nn.Module, blocks: str, start: int) -> None:
    """Train LayerNorms from block start on and after the blocks, not the embeddings.

    blocks is the module path of the block list, e.g. encoder.layers.
    """
    for name, module in backbone.named_modules():
        if not isinstance(module, nn.LayerNorm) or name.startswith("embeddings"):
            continue
        if name.startswith(f"{blocks}."):
            if int(name[len(blocks) + 1 :].split(".")[0]) >= start:
                module.requires_grad_(True)
        else:
            module.requires_grad_(True)


class FrozenSigLIP(FrozenEncoder):
    """SigLIP vision tower (transformers); every output token is a patch.

    With layer set, tokens are read after that block, before the final norm. The pooled
    output is the attention-pooling head on the final normed tokens.
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
        from transformers import SiglipVisionModel

        super().__init__(SiglipVisionModel.from_pretrained(local_path or model_id), chunk, d_feature, n_tokens)
        self.layer = layer
        self.depth = len(self.backbone.encoder.layers)
        self.n_prefix = 0

    def hidden(self, pixels: torch.Tensor, depth: int) -> tuple[torch.Tensor, Any]:
        """Run the first depth blocks; return every token and no positional context."""
        # Mirrors SiglipVisionModel.forward, stopping after block depth.
        x = self.backbone.embeddings(pixels)
        for block in self.backbone.encoder.layers[:depth]:
            x = block(x, None)
        return x, None

    def finish(self, x: torch.Tensor, context: Any, start: int) -> torch.Tensor:
        """Run blocks from start, the final norm and the attention-pooling head."""
        del context
        for block in self.backbone.encoder.layers[start:]:
            x = checkpoint(block, x, None, use_reentrant=False)
        return self.backbone.head(self.backbone.post_layernorm(x))

    def tune_layernorms(self, start: int) -> None:
        """Train LayerNorms from block start on, plus the final norm and the head."""
        tune_late_layernorms(self.backbone, "encoder.layers", start)

    @property
    def layers(self) -> nn.ModuleList:
        """The transformer blocks, in order."""
        return self.backbone.encoder.layers

    def embed(self, pixels: torch.Tensor, pre_norm: nn.Module | None = None) -> tuple[torch.Tensor, Any]:
        """Tokens before block 1 and the block context; SigLIP has no pre-norm."""
        del pre_norm
        return self.backbone.embeddings(pixels), None

    @staticmethod
    def block_inputs(x: torch.Tensor, context: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Positional and keyword arguments of one block call."""
        del context
        return (x, None), {}

    def tokens(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return the last hidden state, or the tokens after block layer."""
        if self.layer is None:
            return self.backbone(pixel_values=pixels).last_hidden_state
        return self.hidden(pixels, self.layer)[0]


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

    @property
    def layers(self) -> nn.ModuleList:
        """The transformer blocks, in order."""
        return self.backbone.blocks

    @property
    def pre_norm(self) -> nn.Module:
        """The norm applied to the embedded tokens before block 1."""
        return self.backbone.norm_pre

    def embed(self, pixels: torch.Tensor, pre_norm: nn.Module | None = None) -> tuple[torch.Tensor, Any]:
        """Tokens before block 1, normed by pre_norm (default norm_pre), and rope."""
        x, rope = self.backbone._pos_embed(self.backbone.patch_embed(pixels))
        return (pre_norm or self.backbone.norm_pre)(x), rope

    @staticmethod
    def block_inputs(x: torch.Tensor, context: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Positional and keyword arguments of one block call."""
        return (x,), {"rope": context}

    def add_lora(self, start: int, rank: int, alpha: float, generator: torch.Generator) -> None:
        """Wrap every linear layer of the blocks from start on in a LoRA update."""
        for block in self.backbone.blocks[start:]:
            for parent, name in ((block.attn, "qkv"), (block.attn, "proj"), (block.mlp, "fc1"), (block.mlp, "fc2")):
                setattr(parent, name, LoRALinear(getattr(parent, name), rank, alpha, generator))

    def tokens(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return patch tokens: final normed tokens, or those after block layer."""
        if self.layer is None:
            return self.backbone.forward_features(pixels)[:, self.n_prefix :]
        return self.hidden(pixels, self.layer)[0][:, self.n_prefix :]


class FrozenDINOv3(FrozenEncoder):
    """DINOv3 (transformers); CLS and register tokens are dropped.

    DINOv3 expects ImageNet normalisation, so pixels are mapped back from [-1, 1].
    """

    mean: torch.Tensor
    std: torch.Tensor

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
        from transformers import AutoModel

        super().__init__(AutoModel.from_pretrained(local_path or model_id), chunk, d_feature, n_tokens)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.n_prefix = 1 + self.backbone.config.num_register_tokens
        self.layer = layer
        self.depth = len(self.backbone.model.layer)

    def normalize(self, pixels: torch.Tensor) -> torch.Tensor:
        """Map [-1, 1] pixels to ImageNet normalisation."""
        return (pixels * 0.5 + 0.5 - self.mean) / self.std

    def hidden(self, pixels: torch.Tensor, depth: int) -> tuple[torch.Tensor, Any]:
        """Run the first depth blocks; return every token and the rotary embedding."""
        # Mirrors DINOv3ViTModel.forward, stopping after block depth.
        pixels = self.normalize(pixels)
        x = self.backbone.embeddings(pixels.to(self.backbone.embeddings.patch_embeddings.weight.dtype))
        rope = self.backbone.rope_embeddings(pixels)
        for block in self.backbone.model.layer[:depth]:
            x = block(x, position_embeddings=rope)
        return x, rope

    def finish(self, x: torch.Tensor, rope: Any, start: int) -> torch.Tensor:
        """Run blocks from start and the final norm; return the class token."""
        for block in self.backbone.model.layer[start:]:
            x = checkpoint(block, x, None, rope, use_reentrant=False)
        return self.backbone.norm(x)[:, 0]

    def tune_layernorms(self, start: int) -> None:
        """Train LayerNorms from block start on, plus the final norm."""
        tune_late_layernorms(self.backbone, "model.layer", start)

    @property
    def layers(self) -> nn.ModuleList:
        """The transformer blocks, in order."""
        return self.backbone.model.layer

    def embed(self, pixels: torch.Tensor, pre_norm: nn.Module | None = None) -> tuple[torch.Tensor, Any]:
        """Tokens before block 1 and the rotary embedding; DINOv3 has no pre-norm."""
        del pre_norm
        pixels = self.normalize(pixels)
        x = self.backbone.embeddings(pixels.to(self.backbone.embeddings.patch_embeddings.weight.dtype))
        return x, self.backbone.rope_embeddings(pixels)

    @staticmethod
    def block_inputs(x: torch.Tensor, context: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Positional and keyword arguments of one block call."""
        return (x,), {"position_embeddings": context}

    def tokens(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return the patch tokens of the last hidden state, or after block layer."""
        if self.layer is None:
            return self.backbone(pixel_values=self.normalize(pixels)).last_hidden_state[:, self.n_prefix :]
        return self.hidden(pixels, self.layer)[0][:, self.n_prefix :]


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


ENCODERS: dict[str, Callable[..., FrozenEncoder]] = {
    "siglip": FrozenSigLIP,
    "pe_core": FrozenPerceptionEncoder,
    "dinov3": FrozenDINOv3,
}


def build_encoder(
    encoder_id: str,
    encoder_path: str | None,
    *,
    chunk: int,
    d_feature: int,
    n_tokens: int,
    image_size: int,
    layer: int | None = None,
) -> nn.Module:
    """Build the frozen encoder named by encoder_id.

    Parameters
    ----------
    encoder_id : str
        stub, or a model id containing siglip, pe_core or dinov3.
    encoder_path : str or None
        Local weights, used instead of downloading encoder_id.
    chunk : int
        Frames encoded per forward call.
    d_feature, n_tokens : int
        Expected feature width and patch-token count.
    image_size : int
        Input side, used only by the stub.
    layer : int or None
        Read tokens after this block; None reads the output.

    Returns
    -------
    nn.Module
        Maps (B, T, 3, S, S) frames to (B, T, n_tokens, d_feature) tokens.
    """
    if encoder_id == "stub":
        return StubEncoder(d_feature, n_tokens, image_size)
    for key, encoder_cls in ENCODERS.items():
        if key in encoder_id.lower():
            extra = {"layer": layer} if layer is not None else {}
            return encoder_cls(encoder_id, encoder_path, chunk, d_feature=d_feature, n_tokens=n_tokens, **extra)
    raise ValueError(f"unknown encoder {encoder_id!r}; expected stub or an id containing {sorted(ENCODERS)}")
