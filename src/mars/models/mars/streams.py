"""Appearance and residual streams, and the fusion that sums them.

Every stream is layer-normalized before its projection, so no stream can dominate
the sum through raw magnitude alone. Streams are summed rather than concatenated,
which keeps activation memory flat in the number of scales.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from mars.models.mars.bandpass import (
    prefix_tables,
    smooth_grid,
    smooth_time,
    smooth_time_masked,
    smoothing_matrix,
)
from mars.models.mars.config import ModelConfig


class ResidualStreams(nn.Module):
    """Fixed band-pass residuals over the patch grid and the frame axis.

    Holds no learnable parameters. Smoothing matrices are non-persistent buffers
    rebuilt from the config, so they never enter a checkpoint.
    """

    def __init__(self, cfg: ModelConfig, n_frames: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.spatial = cfg.streams.spatial.bands
        self.temporal = cfg.streams.temporal.bands
        self.masked = cfg.temporal_validity
        self._space_taus = sorted({t for pair in self.spatial for t in pair})
        self._time_taus = sorted({t for pair in self.temporal for t in pair})
        kind = cfg.streams.operator
        for index, tau in enumerate(self._space_taus):
            self.register_buffer(
                f"space_{index}",
                smoothing_matrix(cfg.grid_side, tau, kind).float(),
                persistent=False,
            )
        for index, tau in enumerate(self._time_taus):
            matrix = prefix_tables(n_frames, tau, kind) if self.masked else smoothing_matrix(n_frames, tau, kind)
            self.register_buffer(f"time_{index}", matrix.float(), persistent=False)

    @property
    def n_streams(self) -> int:
        """Number of residual tensors this module emits."""
        return self.cfg.streams.n_residual

    def _space(self, tau: float) -> torch.Tensor:
        return getattr(self, f"space_{self._space_taus.index(tau)}")

    def _time(self, tau: float) -> torch.Tensor:
        return getattr(self, f"time_{self._time_taus.index(tau)}")

    def _smooth_time(self, x: torch.Tensor, tau: float, valid: torch.Tensor | None) -> torch.Tensor:
        if self.masked and valid is not None:
            return smooth_time_masked(x, self._time(tau), valid)
        return smooth_time(x, self._time(tau))

    def _difference(self, x: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        """Backward difference anchored at the current frame, slot 0 left zero."""
        zero = torch.zeros_like(x[:, :1])
        difference = torch.cat([zero, x[:, 1:] - x[:, :-1]], dim=1)
        if valid is not None:
            edge = (valid[:, :-1] & valid[:, 1:]).to(dtype=x.dtype)[:, :, None, None]
            difference = torch.cat([zero, edge * difference[:, 1:]], dim=1)
        return difference

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Return one tensor per residual stream, each shaped like x.

        The order matches StreamsConfig.residual_names: spatial bands, then
        temporal bands, then the first difference.
        """
        streams: list[torch.Tensor] = []
        for low, high in self.spatial:
            streams.append(
                smooth_grid(x, self._space(low), self.cfg.grid_side)
                - smooth_grid(x, self._space(high), self.cfg.grid_side)
            )
        for low, high in self.temporal:
            streams.append(self._smooth_time(x, low, valid) - self._smooth_time(x, high, valid))
        if self.cfg.streams.first_difference:
            streams.append(self._difference(x, valid))
        return streams


class StreamFusion(nn.Module):
    """Project the appearance stream and every residual stream, then sum them."""

    def __init__(self, cfg: ModelConfig, n_frames: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_norm = nn.LayerNorm(cfg.d_feature)
        self.appearance_proj: nn.Module | None = None
        if cfg.streams.appearance:
            self.appearance_proj = nn.Linear(cfg.d_feature, cfg.d_model)
        self.streams: ResidualStreams | None = None
        self.resid_proj: nn.Module = nn.Identity()
        self.stream_norms = nn.ModuleList()
        self.stream_projs = nn.ModuleList()
        if cfg.streams.n_residual:
            self.streams = ResidualStreams(cfg, n_frames)
            self.resid_proj = nn.Linear(cfg.d_feature, cfg.d_resid)
            for _ in range(cfg.streams.n_residual):
                self.stream_norms.append(nn.LayerNorm(cfg.d_resid))
                self.stream_projs.append(nn.Linear(cfg.d_resid, cfg.d_model))
        self.stream_rms: dict[str, float] = {}

    def _residual_sum(self, resid: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        assert self.streams is not None
        names = self.cfg.streams.residual_names
        out = None
        rms: dict[str, float] = {}
        for name, stream, norm, proj in zip(names, self.streams(resid, valid), self.stream_norms, self.stream_projs):
            projected = proj(norm(stream))
            rms[name] = float(projected.detach().float().norm() / projected.numel() ** 0.5)
            out = projected if out is None else out + projected
        self.stream_rms = rms
        assert out is not None
        return out

    def forward(self, features: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        """(B, T, N, d_feature) encoder features to (B, T, N, d_model) tokens.

        The channel projection runs before the filters. Filters touch only the
        token and frame axes, so projecting first is equivalent to projecting
        afterwards and costs several times less memory.
        """
        normalized = self.patch_norm(features)
        fused = None
        if self.appearance_proj is not None:
            fused = self.appearance_proj(normalized)
        if self.streams is None:
            self.stream_rms = {}
            assert fused is not None
            return fused

        if not self.streams.masked:
            valid = None
        resid = self.resid_proj(normalized)
        if self.cfg.grad_checkpoint and self.training:
            residual = checkpoint(self._residual_sum, resid, valid, use_reentrant=False)
        else:
            residual = self._residual_sum(resid, valid)
        return residual if fused is None else fused + residual
