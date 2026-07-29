"""Cross-attention A2V / V2A imputation heads.

Audio (``N_a``) and visual (``N_v``) tokens stay at their native per-clip
counts, no pooling — imputation runs per-sample rather than over a padded
batched tensor.
"""

from __future__ import annotations

import torch
from torch import nn


class MaskedQueryEmbedding(nn.Module):
    """Query for masked-token reconstruction: mask token + time bucket.

    Independent of the ground-truth token being reconstructed, unlike a
    naive "self as query" approach — see
    ``_CrossModalAttention.forward_masked`` for why that leaks the answer.
    A lookup table, not a sinusoidal encoding: clip durations here are
    bounded and known in advance, so there's no extrapolation need that
    sinusoidal encodings exist for.

    Examples
    --------
    >>> import torch
    >>> emb = MaskedQueryEmbedding(hidden_dim=4, bucket_seconds=1.0, max_seconds=2.0)
    >>> emb(torch.tensor([0.5, 5.0])).shape
    torch.Size([2, 4])
    """

    def __init__(self, hidden_dim: int, bucket_seconds: float, max_seconds: float) -> None:
        super().__init__()
        self.bucket_seconds = bucket_seconds
        self.num_buckets = int(max_seconds // bucket_seconds) + 1
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))
        self.time_embedding = nn.Embedding(self.num_buckets, hidden_dim)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Return ``(n, hidden_dim)`` queries for ``timestamps`` ``(n,)``.

        Out-of-range timestamps clamp into the final bucket rather than
        erroring.
        """
        bucket_idx = torch.div(timestamps, self.bucket_seconds, rounding_mode="floor").long()
        bucket_idx = bucket_idx.clamp(0, self.num_buckets - 1)
        return self.mask_token + self.time_embedding(bucket_idx)


class _CrossModalAttention(nn.Module):
    """One directional cross-attention block: target queries, source keys/values.

    Optionally biased by timestamp proximity — an additive term that decays
    with ``|t_q - t_k|`` — nudging attention toward temporally-nearby source
    tokens without hard-constraining it, since audio/visual token rates
    differ (13 Hz vs. one timestamp per merged spatial tile) and a rigid
    alignment window would need per-token-rate-specific tuning.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        use_timestamp_bias: bool,
        mask_query_bucket_seconds: float,
        mask_query_max_seconds: float,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.timestamp_decay = nn.Parameter(torch.tensor(1.0)) if use_timestamp_bias else None
        self.masked_query = MaskedQueryEmbedding(hidden_dim, mask_query_bucket_seconds, mask_query_max_seconds)

    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        target_timestamps: torch.Tensor,
        source_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attend ``target`` queries over ``source`` keys/values.

        Parameters
        ----------
        target : torch.Tensor
            ``(n_target, hidden_dim)``.
        source : torch.Tensor
            ``(n_source, hidden_dim)``.

        Returns
        -------
        torch.Tensor
            ``(n_target, hidden_dim)``.

        Used directly by the unmasked/inference path, where ``target`` is a
        real token being enriched with context, not reconstructed. See
        ``forward_masked`` for the self-supervised reconstruction path.
        """
        attn_bias = None
        if self.timestamp_decay is not None:
            time_gap = (target_timestamps.unsqueeze(1) - source_timestamps.unsqueeze(0)).abs()
            attn_bias = (-nn.functional.softplus(self.timestamp_decay) * time_gap).to(target.dtype)
        out, _ = self.attn(
            target.unsqueeze(0),
            source.unsqueeze(0),
            source.unsqueeze(0),
            attn_mask=attn_bias,
            need_weights=False,
        )
        return out.squeeze(0)

    def forward_masked(
        self,
        source: torch.Tensor,
        source_timestamps: torch.Tensor,
        target_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct masked tokens from position + visible source only.

        Queries come from ``masked_query`` (mask token + timestamp bucket),
        never from the ground-truth tokens being reconstructed — passing
        the true value in as the query would leak the reconstruction target
        into the attention weighting itself, contaminating the calibrated
        uncertainty this module exists to produce.
        """
        query = self.masked_query(target_timestamps)
        return self.forward(query, source, target_timestamps, source_timestamps)


class ImputationModule(nn.Module):
    """Owns both cross-modal directions (A2V, V2A) and the masked/unmasked dispatch."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        use_timestamp_bias: bool,
        mask_query_bucket_seconds: float,
        mask_query_max_seconds: float,
    ) -> None:
        super().__init__()
        self.a2v = _CrossModalAttention(
            hidden_dim, num_heads, use_timestamp_bias, mask_query_bucket_seconds, mask_query_max_seconds
        )
        self.v2a = _CrossModalAttention(
            hidden_dim, num_heads, use_timestamp_bias, mask_query_bucket_seconds, mask_query_max_seconds
        )

    def forward(
        self,
        z_audio: torch.Tensor,
        z_visual: torch.Tensor,
        audio_timestamps: torch.Tensor,
        visual_timestamps: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
        visual_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fused ``(z_audio', z_visual')``, same shapes as the inputs.

        If ``audio_mask``/``visual_mask`` are given (``True`` = masked),
        masked tokens are replaced with cross-modal predictions from the
        other modality's visible tokens; visible tokens pass through
        unchanged. If both are omitted, every token is enriched with
        cross-modal context instead (see ``_forward_unmasked``). They must
        both be given or both omitted.
        """
        if audio_mask is None and visual_mask is None:
            return self._forward_unmasked(z_audio, z_visual, audio_timestamps, visual_timestamps)
        if audio_mask is None or visual_mask is None:
            raise ValueError("audio_mask and visual_mask must both be given, or both omitted")
        return self._forward_masked(z_audio, z_visual, audio_timestamps, visual_timestamps, audio_mask, visual_mask)

    def _forward_unmasked(
        self,
        z_audio: torch.Tensor,
        z_visual: torch.Tensor,
        audio_timestamps: torch.Tensor,
        visual_timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add cross-modal context to every token, residual-style.

        Every audio token attends over the full visual stream and vice
        versa; the result is added to the original token rather than
        concatenated, since both streams already share ``hidden_dim`` and
        concatenation would just need a projection back down anyway.
        """
        v2a_context = self.v2a(z_audio, z_visual, audio_timestamps, visual_timestamps)
        a2v_context = self.a2v(z_visual, z_audio, visual_timestamps, audio_timestamps)
        return z_audio + v2a_context, z_visual + a2v_context

    def _forward_masked(
        self,
        z_audio: torch.Tensor,
        z_visual: torch.Tensor,
        audio_timestamps: torch.Tensor,
        visual_timestamps: torch.Tensor,
        audio_mask: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct masked tokens of each modality from the other's visible ones."""
        z_audio_fused = z_audio.clone()
        z_visual_fused = z_visual.clone()

        visible_visual = z_visual[~visual_mask]
        visible_visual_ts = visual_timestamps[~visual_mask]
        if audio_mask.any() and visible_visual.shape[0] > 0:
            masked_audio_ts = audio_timestamps[audio_mask]
            z_audio_fused[audio_mask] = self.v2a.forward_masked(visible_visual, visible_visual_ts, masked_audio_ts)

        visible_audio = z_audio[~audio_mask]
        visible_audio_ts = audio_timestamps[~audio_mask]
        if visual_mask.any() and visible_audio.shape[0] > 0:
            masked_visual_ts = visual_timestamps[visual_mask]
            z_visual_fused[visual_mask] = self.a2v.forward_masked(visible_audio, visible_audio_ts, masked_visual_ts)

        return z_audio_fused, z_visual_fused
