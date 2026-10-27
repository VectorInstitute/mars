"""X-CLIP vision tower whose cross-frame message attention skips padded frames.

Mirrors Hugging Face's XCLIPVisionModel, parameter names included, so the released
weights load unchanged. Each block first lets the class tokens of a group of
consecutive frames exchange messages, then runs per-frame attention with that
frame's message appended. Padded frames are masked out as message keys, so
genuine frames never read from them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class XCLIPAttention(nn.Module):
    """Multi-head attention with separate query, key and value projections."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor | None = None) -> torch.Tensor:
        """Attend over (N, L, D) tokens; allowed is a (N, L, L) boolean mask or None."""
        batch, length, width = x.shape

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(batch, length, self.heads, -1).transpose(1, 2)

        mask = None if allowed is None else allowed[:, None]
        out = F.scaled_dot_product_attention(split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x)), mask)
        return self.out_proj(out.transpose(1, 2).reshape(batch, length, width))


class XCLIPMLP(nn.Module):
    """Two linear layers with QuickGELU between them."""

    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fc2(QuickGELU(fc1(x)))."""
        x = self.fc1(x)
        return self.fc2(x * torch.sigmoid(1.702 * x))


class XCLIPLayer(nn.Module):
    """Cross-frame message exchange, then per-frame attention and MLP."""

    def __init__(self, width: int, heads: int, hidden: int, group: int) -> None:
        super().__init__()
        self.group = group
        self.message_fc = nn.Linear(width, width)
        self.message_ln = nn.LayerNorm(width)
        self.message_attn = XCLIPAttention(width, heads)
        self.self_attn = XCLIPAttention(width, heads)
        self.layer_norm1 = nn.LayerNorm(width)
        self.mlp = XCLIPMLP(width, hidden)
        self.layer_norm2 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        """(G * group, L, D) frames; allowed (G, group, group) masks message keys."""
        frames, length, width = x.shape
        message = self.message_fc(x[:, 0]).view(frames // self.group, self.group, width)
        message = message + self.message_attn(self.message_ln(message), allowed)
        x = torch.cat([x, message.reshape(frames, 1, width)], dim=1)
        x = x + self.self_attn(self.layer_norm1(x))
        x = x[:, :length]
        return x + self.mlp(self.layer_norm2(x))


class XCLIPEmbeddings(nn.Module):
    """Patch embedding, class token and learned positions."""

    def __init__(self, image_size: int, patch_size: int, width: int) -> None:
        super().__init__()
        self.class_embedding = nn.Parameter(torch.randn(width))
        self.patch_embedding = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.position_embedding = nn.Embedding((image_size // patch_size) ** 2 + 1, width)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """(N, 3, S, S) to (N, 1 + L, D) tokens."""
        x = self.patch_embedding(pixels).flatten(2).transpose(1, 2)
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        return torch.cat([cls, x], dim=1) + self.position_embedding.weight.to(x.dtype)


class XCLIPEncoder(nn.Module):
    """The block stack, named as in the released weights."""

    def __init__(self, width: int, layers: int, heads: int, hidden: int, group: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(XCLIPLayer(width, heads, hidden, group) for _ in range(layers))


class XCLIPVisionTransformer(nn.Module):
    """X-CLIP ViT over groups of consecutive frames.

    Parameters
    ----------
    image_size, patch_size, width, layers, heads, hidden : int
        ViT geometry; X-CLIP B/16 is 224, 16, 768, 12, 12, 3072.
    group : int
        Consecutive frames that exchange messages; 8 as pretrained.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        hidden: int = 3072,
        group: int = 8,
    ) -> None:
        super().__init__()
        self.group = group
        self.embeddings = XCLIPEmbeddings(image_size, patch_size, width)
        self.pre_layernorm = nn.LayerNorm(width)
        self.encoder = XCLIPEncoder(width, layers, heads, hidden, group)
        self.post_layernorm = nn.LayerNorm(width)

    def forward(self, frames: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode (B, T, 3, S, S) frames; T must be a multiple of group.

        Returns the last hidden state (B, T, 1 + L, D) and the pooled class token
        (B, T, D). A padded frame reads only genuine frames and itself; no genuine
        frame reads a padded one.
        """
        batch, n_frames = frames.shape[:2]
        if n_frames % self.group:
            raise ValueError(f"{n_frames} frames is not a multiple of the message group {self.group}")
        keys = valid.to(device=frames.device, dtype=torch.bool).reshape(-1, 1, self.group)
        allowed = keys | torch.eye(self.group, dtype=torch.bool, device=frames.device)
        x = self.pre_layernorm(self.embeddings(frames.flatten(0, 1)))
        for layer in self.encoder.layers:
            x = checkpoint(layer, x, allowed, use_reentrant=False) if self.training else layer(x, allowed)
        pooled = self.post_layernorm(x[:, 0])
        return x.reshape(batch, n_frames, *x.shape[1:]), pooled.reshape(batch, n_frames, -1)


def load_xclip_weights(tower: XCLIPVisionTransformer, model_id: str) -> None:
    """Load a released X-CLIP checkpoint's vision weights by Hugging Face id or folder.

    The checkpoint stores the full X-CLIP model; its vision_model entries map onto
    this tower by name.
    """
    from pathlib import Path

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    local = Path(model_id) / "model.safetensors"
    path = local if local.exists() else hf_hub_download(model_id, "model.safetensors")
    prefix = "vision_model."
    state = {
        k.removeprefix(prefix): v
        for k, v in load_file(str(path)).items()
        if k.startswith(prefix) and not k.endswith("position_ids")
    }
    tower.load_state_dict(state)
