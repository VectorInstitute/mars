"""Exact per-frame and per-patch contributions to the MARS decision.

After the temporal transformer the residual path averages over valid frames, pools
patches with fixed weights and applies a linear head, so its margin is a sum of one
term per frame and patch. The semantic margin is a mean of per-frame margins, and at
inference the standardisation and gate are affine in the two margins. The fused
margin is therefore an exact sum of frame and patch contributions plus a constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from mars.models.mars.model import Mars
from mars.models.mars.semantic_path import frame_subset


@dataclass
class Attribution:
    """Contributions for a batch of clips; margins are fake minus real.

    Attributes
    ----------
    margin : torch.Tensor
        (B,) fused margin of the model's own forward pass.
    margin_from_parts : torch.Tensor
        (B,) the same margin rebuilt from the contributions below.
    frame_scores : torch.Tensor
        (B, T) contribution of each frame to the fused margin; zero for padded frames.
    frame_maps : torch.Tensor
        (B, T, N) residual contribution of each frame and patch to the fused margin.
    residual_terms : torch.Tensor
        (B, T, N) residual contributions c to the residual margin, before fusion.
    semantic_margins : torch.Tensor
        (B, S) margin of each sampled frame in the semantic path.
    semantic_index : torch.Tensor
        (B, S) frame index of each sampled frame.
    patch_weights : torch.Tensor
        (B, N) attention-pool weights of the residual path.
    gate : torch.Tensor
        (B,) gate weight w of the semantic path.
    constant : torch.Tensor
        (B,) part of the fused margin not owed to any frame.
    """

    margin: torch.Tensor
    margin_from_parts: torch.Tensor
    frame_scores: torch.Tensor
    frame_maps: torch.Tensor
    residual_terms: torch.Tensor
    semantic_margins: torch.Tensor
    semantic_index: torch.Tensor
    patch_weights: torch.Tensor
    gate: torch.Tensor
    constant: torch.Tensor


def _margin_row(weight: torch.Tensor) -> torch.Tensor:
    """Fake-minus-real difference of a two-class weight or bias."""
    return weight[1] - weight[0]


def _running_stats(norm: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the running mean and 1 / sqrt(variance + eps) of a margin BatchNorm."""
    mean, var = norm.running_mean, norm.running_var
    if not isinstance(mean, torch.Tensor) or not isinstance(var, torch.Tensor):
        raise ValueError("margin standardisation has no running statistics")
    return mean.float(), 1.0 / torch.sqrt(var.float() + float(norm.eps))  # type: ignore[arg-type]


@torch.no_grad()
def attribute(model: Mars, frames: torch.Tensor, valid: torch.Tensor) -> Attribution:
    """Run two-path MARS once and split its fused margin over frames and patches.

    Parameters
    ----------
    model : Mars
        Two-path MARS in eval mode, so margin standardisation uses running statistics.
    frames : torch.Tensor
        (B, T, 3, S, S) clip frames.
    valid : torch.Tensor
        (B, T) mask of genuine frames, a prefix per clip.

    Returns
    -------
    Attribution
        Contributions and the margins they sum to.
    """
    if model.training:
        raise ValueError("attribute needs the model in eval mode")
    if model.residual is None or model.semantic is None or model.gate is None or model.margin_norms is None:
        raise ValueError("attribute needs two-path MARS")

    captured: dict[str, Any] = {}
    hooks = [
        model.residual.transformer.register_forward_hook(lambda m, i, o: captured.__setitem__("hidden", o[0])),
        model.residual.register_forward_hook(lambda m, i, o: captured.__setitem__("weights", o.patch_weights)),
        model.semantic.register_forward_hook(lambda m, i, o: captured.__setitem__("pooled", i[0])),
        model.gate.register_forward_hook(lambda m, i, o: captured.__setitem__("gate", o)),
    ]
    try:
        out = model(frames, valid=valid)
    finally:
        for hook in hooks:
            hook.remove()

    batch, n_frames = frames.shape[:2]
    mask = valid.to(device=frames.device, dtype=torch.float32)
    n_valid = mask.sum(dim=1)

    # Residual path: c[t, n] = m_t alpha_n / T_v * omega^T h[t, n].
    hidden = captured["hidden"].float().reshape(batch, -1, n_frames, captured["hidden"].shape[-1])
    alpha = captured["weights"].float()
    classifier = model.residual.classifier
    omega = _margin_row(classifier.weight.float())
    beta = _margin_row(classifier.bias.float())
    projected = torch.einsum("bntd,d->btn", hidden, omega)
    residual_terms = projected * alpha[:, None, :] * (mask / n_valid[:, None])[:, :, None]

    # Semantic path: one margin per sampled frame, averaged.
    unit = F.normalize(captured["pooled"].float(), dim=-1)
    semantic_margins = (unit @ _margin_row(model.semantic.classifier.weight.float())) + _margin_row(
        model.semantic.classifier.bias.float()
    )
    semantic_index = frame_subset(valid, model.cfg.semantic_frames)
    n_semantic = semantic_margins.shape[1]

    # Standardisation and gate, as in Mars.forward (eval: running statistics).
    mean_sem, scale_sem = _running_stats(model.margin_norms[0])
    mean_res, scale_res = _running_stats(model.margin_norms[1])
    w = torch.sigmoid(captured["gate"].float())[:, 0]

    frame_maps = ((1 - w) * scale_res)[:, None, None] * residual_terms
    semantic_terms = (w * scale_sem)[:, None] * semantic_margins / n_semantic
    frame_scores = frame_maps.sum(dim=2)
    frame_scores = frame_scores.scatter_add(1, semantic_index, semantic_terms)

    constant = w * scale_sem * (-mean_sem) + (1 - w) * scale_res * (beta - mean_res)
    logits = out.logits.float()
    return Attribution(
        margin=logits[:, 1] - logits[:, 0],
        margin_from_parts=frame_scores.sum(dim=1) + constant,
        frame_scores=frame_scores,
        frame_maps=frame_maps,
        residual_terms=residual_terms,
        semantic_margins=semantic_margins,
        semantic_index=semantic_index,
        patch_weights=alpha,
        gate=w,
        constant=constant,
    )
