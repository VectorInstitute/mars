"""``AVRModel``: composes imputation, experts, routing, and fusion into one pass."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from deepfake_moe.models.avr.config import AVRConfig
from deepfake_moe.models.avr.experts import ForensicExpert
from deepfake_moe.models.avr.fusion import AggregateProject, InverseVarianceFusion
from deepfake_moe.models.avr.imputation import ImputationModule
from deepfake_moe.models.avr.router import TopKRouter


@dataclass
class AVROutput:
    """Per-token outputs of one ``AVRModel`` forward pass, split out by modality."""

    evidence_audio: torch.Tensor  # (N_a, hidden_dim)
    evidence_visual: torch.Tensor  # (N_v, hidden_dim)
    mu_star: torch.Tensor  # (N_a + N_v, hidden_dim)
    log_sigma_star_sq: torch.Tensor  # (N_a + N_v, 1)
    aux_loss: torch.Tensor


class AVRModel(nn.Module):
    """Runs the frozen encoders' token streams through the full AVR pipeline.

    All experts are evaluated on every token, then the router's top-k
    indices gather each token's selected experts before fusion. This is
    simple dense compute rather than sparse dispatch, which is fine at the
    small expert counts (4-5) this module targets; if ``num_experts`` grows
    much larger, a proper grouped/sparse dispatch would be worth adding.
    """

    def __init__(self, config: AVRConfig) -> None:
        super().__init__()
        self.config = config
        self.imputation = ImputationModule(
            config.hidden_dim,
            config.imputation_num_heads,
            config.imputation_timestamp_bias,
            config.mask_query_bucket_seconds,
            config.mask_query_max_seconds,
        )
        self.experts = nn.ModuleList(
            [ForensicExpert(config.hidden_dim, config.expert_ffn_dim) for _ in range(config.num_experts)]
        )
        self.router = TopKRouter(config.hidden_dim, config.num_experts, config.top_k)
        self.fusion = InverseVarianceFusion()
        self.aggregate_project = AggregateProject(config.hidden_dim, config.hidden_dim)

    def forward(
        self,
        z_audio: torch.Tensor,
        z_visual: torch.Tensor,
        audio_timestamps: torch.Tensor,
        visual_timestamps: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
        visual_mask: torch.Tensor | None = None,
    ) -> AVROutput:
        """Run one forward pass over a single sample's audio/visual token streams.

        ``z_audio``/``z_visual`` are the frozen encoders' native-length
        token streams (see ``encoders.py``). ``audio_mask``/``visual_mask``
        are only given during masked self-supervised pretraining; otherwise
        both are omitted and imputation runs in its always-on residual mode.
        """
        z_audio_fused, z_visual_fused = self.imputation(
            z_audio, z_visual, audio_timestamps, visual_timestamps, audio_mask, visual_mask
        )
        n_audio = z_audio.shape[0]
        tokens = torch.cat([z_audio_fused, z_visual_fused], dim=0)

        # `topk_weight` (the router's own confidence) is intentionally unused
        # here — see fusion.py for why reliability, not relevance, decides
        # the blend.
        _topk_weight, topk_indices, aux_loss = self.router(tokens)

        expert_outputs = [expert(tokens) for expert in self.experts]
        stacked_mu = torch.stack([mu for mu, _ in expert_outputs], dim=1)  # (N, num_experts, hidden_dim)
        stacked_log_sigma_sq = torch.stack([lsq for _, lsq in expert_outputs], dim=1)  # (N, num_experts, 1)

        mu_idx = topk_indices.unsqueeze(-1).expand(-1, -1, stacked_mu.shape[-1])
        selected_mu = torch.gather(stacked_mu, 1, mu_idx)  # (N, top_k, hidden_dim)
        selected_log_sigma_sq = torch.gather(stacked_log_sigma_sq, 1, topk_indices.unsqueeze(-1))  # (N, top_k, 1)

        mu_star, log_sigma_star_sq = self.fusion(selected_mu, selected_log_sigma_sq)
        evidence_tokens = self.aggregate_project(mu_star, log_sigma_star_sq)

        return AVROutput(
            evidence_audio=evidence_tokens[:n_audio],
            evidence_visual=evidence_tokens[n_audio:],
            mu_star=mu_star,
            log_sigma_star_sq=log_sigma_star_sq,
            aux_loss=aux_loss,
        )
