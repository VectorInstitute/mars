"""Top-k softmax gating over Forensic Experts, with a load-balancing auxiliary loss.

The router only picks which experts run; it does not weight their final
blend (see ``fusion.py``) — its gate weights feed only the auxiliary loss.
"""

from __future__ import annotations

import torch
from torch import nn


def load_balancing_loss(router_probs: torch.Tensor, selected_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Switch-Transformer-style load-balancing loss.

    ``router_probs``: ``(N, num_experts)`` full softmax distribution.
    ``selected_experts``: ``(N, top_k)`` indices actually dispatched.
    Encourages the fraction of tokens routed to each expert and that
    expert's average routing probability to be uncorrelated and uniform
    across experts — a standard MoE anti-collapse regularizer.
    """
    n_tokens, top_k = selected_experts.shape
    one_hot = nn.functional.one_hot(selected_experts, num_experts).float()  # (N, top_k, E)
    tokens_per_expert = one_hot.sum(dim=(0, 1)) / (n_tokens * top_k)  # f_e, (E,)
    prob_per_expert = router_probs.mean(dim=0)  # P_e, (E,)
    return num_experts * torch.sum(tokens_per_expert * prob_per_expert)


class TopKRouter(nn.Module):
    """Gates ``(N, hidden_dim)`` tokens to their top-``k`` experts."""

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(topk_weight, topk_indices, aux_loss)``.

        ``topk_weight``/``topk_indices`` are ``(N, top_k)``. ``topk_weight``
        is renormalized to sum to 1 across the selected experts, but isn't
        consumed by fusion (see module docstring) — logging/inspection only.
        """
        logits = self.gate(tokens)
        probs = nn.functional.softmax(logits, dim=-1)
        topk_weight, topk_indices = probs.topk(self.top_k, dim=-1)
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        aux_loss = load_balancing_loss(probs, topk_indices, self.num_experts)
        return topk_weight, topk_indices, aux_loss
