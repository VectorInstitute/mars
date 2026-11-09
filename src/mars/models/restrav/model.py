"""ReStraV network: frozen DINOv2 trajectory geometry and a small MLP.

Each genuine frame is encoded by DINOv2 ViT-S/14 at 224; its normalised class and
patch tokens, flattened, are a point on the clip's trajectory. The trajectory's
step lengths and turning angles give 21 features, standardised with training-set
statistics and classified by an MLP into one score.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mars.models.restrav.config import ModelConfig
from mars.models.restrav.geometry import trajectory_features


@dataclass
class ReStraVOutput:
    """Forward-pass tensors: (B, 2) logits, real fixed at 0, fake the score."""

    logits: torch.Tensor


class RunningStandardiser(nn.Module):
    """Standardise features with the mean and variance of every training sample seen.

    Stands in for the released pre-pass over the training set: statistics
    accumulate in training and are frozen in evaluation.
    """

    mean: torch.Tensor
    var: torch.Tensor
    count: torch.Tensor

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Update the statistics in training, then standardise x."""
        if self.training:
            with torch.no_grad():
                n = float(len(x))
                total = self.count + n
                batch_mean = x.mean(dim=0)
                shift = batch_mean - self.mean
                new_mean = self.mean + shift * n / total
                spread = self.var * self.count + x.var(dim=0, unbiased=False) * n + shift**2 * self.count * n / total
                self.mean.copy_(new_mean)
                self.var.copy_(spread / total)
                self.count.copy_(total)
        return (x - self.mean) / (self.var.sqrt() + self.eps)


class ReStraV(nn.Module):
    """Frozen DINOv2, trajectory features, standardisation and an MLP."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = torch.hub.load(cfg.encoder_repo, cfg.encoder_id, trust_repo=True)  # type: ignore[no-untyped-call]
        self.encoder.eval().requires_grad_(False)
        n_features = cfg.early_steps + cfg.early_angles + 8
        self.standardise = RunningStandardiser(n_features)
        dims = [n_features, *cfg.hidden]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], 1)

    @property
    def classifier(self) -> nn.Linear:
        """The final linear layer."""
        return self.head

    def train(self, mode: bool = True) -> ReStraV:
        """Train the MLP and statistics; DINOv2 stays in eval."""
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def embed(self, pixels: torch.Tensor) -> torch.Tensor:
        """Map (N, 3, S, S) frames in [-1, 1] to flattened normalised tokens.

        As released: pixels in [0, 1] without mean and std, and float32 throughout,
        since the features are distances between these long vectors.
        """
        with torch.autocast(pixels.device.type, enabled=False):
            tokens = self.encoder.forward_features(pixels.float() * 0.5 + 0.5)
        return torch.cat([tokens["x_norm_clstoken"][:, None], tokens["x_norm_patchtokens"]], dim=1).flatten(1)

    def forward(self, frames: torch.Tensor, valid: torch.Tensor) -> ReStraVOutput:
        """Classify (B, T, 3, S, S) frames; valid (B, T) marks genuine frames."""
        batch, n_frames = frames.shape[:2]
        valid = valid.to(device=frames.device, dtype=torch.bool)
        genuine = frames.flatten(0, 1)[valid.flatten()]
        chunk = self.cfg.encoder_chunk
        points = torch.cat([self.embed(genuine[i : i + chunk]) for i in range(0, len(genuine), chunk)]).float()
        z = points.new_zeros(batch, n_frames, points.shape[-1])
        z[valid] = points
        features = trajectory_features(z, valid, self.cfg.early_steps, self.cfg.early_angles)
        score = self.head(self.mlp(self.standardise(features)))
        return ReStraVOutput(logits=torch.cat([torch.zeros_like(score), score], dim=1))


def build_model(cfg: ModelConfig) -> ReStraV:
    """Construct ReStraV with the released DINOv2 from torch.hub."""
    return ReStraV(cfg)
