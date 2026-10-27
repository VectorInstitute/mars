"""MARS: residual and semantic paths over one frozen encoder, read two ways.

The residual path runs the multiscale streams and temporal trunk on frozen patch
tokens after block encoder_layer. The semantic path runs its frames through the
remaining blocks, with only their LayerNorms trainable, and classifies each frame
from the encoder's pooled output on the unit sphere. With semantic_early_norms it
re-encodes its frames from pixels with its own trainable copies of the LayerNorms
before the split; otherwise it continues the residual path's frozen tokens. Each
path's margin is standardised, then a per-clip gate mixes them. The fused loss
trains only the gate.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint

from mars.models.mars.config import ModelConfig
from mars.models.mars.encoder import FrozenDINOv3, FrozenPerceptionEncoder, FrozenSigLIP, build_encoder
from mars.models.mars.residual_path import MarsOutput, ResidualPath
from mars.models.mars.semantic_path import SemanticPath, frame_subset
from mars.models.mars.streams import StreamFusion


class PassThrough(nn.Module):
    """Stands in for an encoder when the tokens are already computed."""

    def __init__(self, d_feature: int, n_tokens: int) -> None:
        super().__init__()
        self.d_feature = d_feature
        self.n_tokens = n_tokens

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return the tokens unchanged."""
        return tokens


class Mars(nn.Module):
    """One frozen encoder; residual and semantic paths; a gate over their margins."""

    def __init__(self, cfg: ModelConfig, n_frames: int, image_size: int = 384) -> None:
        super().__init__()
        self.cfg = cfg
        encoder = build_encoder(
            cfg.encoder_id,
            cfg.encoder_path,
            chunk=cfg.encoder_chunk,
            d_feature=cfg.d_feature,
            n_tokens=cfg.n_tokens,
            image_size=image_size,
        )
        if not isinstance(encoder, (FrozenPerceptionEncoder, FrozenSigLIP, FrozenDINOv3)):
            raise ValueError("Mars needs an encoder that can be split: Perception Encoder, SigLIP or DINOv3")
        self.encoder = encoder
        self.split = cfg.encoder_layer if cfg.encoder_layer is not None else encoder.depth
        self.residual = (
            ResidualPath(cfg, n_frames, image_size, PassThrough(cfg.d_feature, cfg.n_tokens))
            if "residual" in cfg.paths
            else None
        )
        self.semantic = SemanticPath(cfg.d_feature, cfg.n_classes) if "semantic" in cfg.paths else None
        if self.semantic is not None and cfg.tune_semantic_norms:
            encoder.tune_layernorms(self.split)
        # Zero init: the gate starts as an even mix of the two paths.
        self.gate = nn.Linear(cfg.d_feature + cfg.d_model, 1) if self.residual and self.semantic else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        # Running statistics put both paths' margins on one scale for the gate.
        self.margin_norms = (
            nn.ModuleList([nn.BatchNorm1d(1, affine=False) for _ in range(2)]) if self.gate is not None else None
        )
        # Built last, from its own generator, so the other modules initialise as before.
        if self.semantic is not None and cfg.semantic_lora_rank > 0 and isinstance(encoder, FrozenPerceptionEncoder):
            generator = torch.Generator().manual_seed(torch.initial_seed())
            encoder.add_lora(self.split, cfg.semantic_lora_rank, cfg.semantic_lora_alpha, generator)
        # Trainable copies of the pre-norm (PE only) and each LayerNorm in blocks
        # 1..split, used only when the semantic path re-encodes its frames; the shared
        # pass keeps the frozen originals. Copies draw no random numbers, so the rest
        # initialises as before.
        self.early_norms: nn.ModuleDict | None = None
        if self.semantic is not None and cfg.semantic_early_norms:
            copies: dict[str, nn.Module] = {}
            if isinstance(encoder, FrozenPerceptionEncoder):
                copies["norm_pre"] = copy.deepcopy(encoder.pre_norm)
            for index, block in enumerate(encoder.layers[: self.split]):
                for name, module in block.named_modules():
                    if isinstance(module, nn.LayerNorm):
                        copies[f"{index}.{name}"] = copy.deepcopy(module)
            self.early_norms = nn.ModuleDict({key.replace(".", "_"): norm for key, norm in copies.items()})
            for parameter in self.early_norms.parameters():
                parameter.requires_grad_(True)

    @property
    def classifier(self) -> nn.Linear:
        """The residual head when present, else the semantic head."""
        return self.residual.classifier if self.residual is not None else self.semantic.classifier  # type: ignore[union-attr]

    @property
    def fusion(self) -> StreamFusion | None:
        """The residual path's stream fusion, for logging."""
        return self.residual.fusion if self.residual is not None else None

    @staticmethod
    def _standardised(norm: nn.Module, logits: torch.Tensor) -> torch.Tensor:
        """Two-class logits whose fake-minus-real margin is standardised by norm."""
        margin = norm((logits[:, 1] - logits[:, 0])[:, None])[:, 0]
        return torch.stack([-margin / 2, margin / 2], dim=1)

    def _early_norm(self, key: str) -> nn.LayerNorm:
        return self.early_norms[key.replace(".", "_")]  # type: ignore[index, return-value]

    def _semantic_tokens(self, pixels: torch.Tensor) -> tuple[torch.Tensor, Any]:
        """Re-encode (N, 3, S, S) frames through blocks 1..split with LayerNorm copies.

        Patch embedding, positions and every attention and MLP weight are the shared
        frozen ones; only the LayerNorm weights and biases are overridden.
        """
        encoder = self.encoder
        pre_norm = self._early_norm("norm_pre") if "norm_pre" in self.early_norms else None  # type: ignore[operator]
        x, context = encoder.embed(pixels, pre_norm)
        for index, block in enumerate(encoder.layers[: self.split]):
            override = {
                f"{name}.{kind}": getattr(self._early_norm(f"{index}.{name}"), kind)
                for name, module in block.named_modules()
                if isinstance(module, nn.LayerNorm)
                for kind in ("weight", "bias")
            }
            x = checkpoint(
                lambda h, blk=block, ovr=override: functional_call(blk, ovr, *encoder.block_inputs(h, context)),
                x,
                use_reentrant=False,
            )
        return x, context

    def forward(self, frames: torch.Tensor, valid: torch.Tensor) -> MarsOutput:
        """Classify (B, T, 3, S, S) frames; path_logits is (semantic, residual)."""
        batch, n_frames = frames.shape[:2]
        flat = frames.flatten(0, 1)
        with torch.no_grad():
            parts = [
                self.encoder.hidden(flat[i : i + self.encoder.chunk], self.split)
                for i in range(0, len(flat), self.encoder.chunk)
            ]
        hidden = torch.cat([h for h, _ in parts])
        rope = parts[0][1]

        logits: dict[str, torch.Tensor] = {}
        features: list[torch.Tensor] = []
        if self.semantic is not None:
            index = frame_subset(valid, self.cfg.semantic_frames)
            rows = torch.arange(batch, device=frames.device)[:, None]
            if self.early_norms is not None:
                chosen, semantic_rope = self._semantic_tokens(frames[rows, index].flatten(0, 1))
            else:
                chosen = hidden.reshape(batch, n_frames, *hidden.shape[1:])[rows, index].flatten(0, 1)
                semantic_rope = rope
            pooled = self.encoder.finish(chosen, semantic_rope, self.split).reshape(batch, index.shape[1], -1)
            embedding, logits["semantic"] = self.semantic(pooled)
            features.append(embedding)
        if self.residual is not None:
            tokens = hidden[:, self.encoder.n_prefix :].reshape(batch, n_frames, self.cfg.n_tokens, self.cfg.d_feature)
            out = self.residual(tokens, valid=valid)
            logits["residual"] = out.logits.float()
            features.append(F.normalize(out.pooled.float(), dim=-1))

        if self.gate is None or self.margin_norms is None:
            return MarsOutput(logits=next(iter(logits.values())))
        # Detached: the fused loss trains only the gate; paths learn from their own.
        semantic, residual = (
            self._standardised(norm, path.detach())
            for norm, path in zip(self.margin_norms, (logits["semantic"], logits["residual"]))
        )
        weight = torch.sigmoid(self.gate(torch.cat(features, dim=-1).detach()))
        logits["fused"] = weight * semantic + (1 - weight) * residual
        return MarsOutput(logits=logits[self.cfg.output], path_logits=(logits["semantic"], logits["residual"]))


def build_model(cfg: ModelConfig, n_frames: int, image_size: int = 384) -> nn.Module:
    """Construct MARS, or single-path MARS when paths is residual only."""
    if cfg.paths == ["residual"]:
        return ResidualPath(cfg, n_frames, image_size)
    return Mars(cfg, n_frames, image_size)
