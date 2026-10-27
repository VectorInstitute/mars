"""Trajectory geometry of per-frame embeddings, from the released dinov2_features.py.

Frames are points z_t; steps are z_(t+1) - z_t. The released features are the first
step lengths and turning angles, and mean, min, max and variance of all of them.
Here every statistic runs over genuine steps only, so padded frames never count.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_geometry(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Step lengths (B, T-1) and turning angles in degrees (B, T-2) of points."""
    delta = z[:, 1:] - z[:, :-1]
    d = delta.norm(dim=-1)
    cos = F.cosine_similarity(delta[:, :-1], delta[:, 1:], dim=-1)
    theta = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))
    return d, theta


def masked_moments(x: torch.Tensor, valid: torch.Tensor) -> list[torch.Tensor]:
    """Mean, min, max and population variance of each row over its valid entries."""
    weight = valid.to(x.dtype)
    count = weight.sum(dim=-1).clamp(min=1)
    mean = (x * weight).sum(dim=-1) / count
    var = ((x - mean[:, None]) ** 2 * weight).sum(dim=-1) / count
    low = x.masked_fill(~valid, float("inf")).amin(dim=-1)
    high = x.masked_fill(~valid, float("-inf")).amax(dim=-1)
    any_valid = valid.any(dim=-1)
    return [mean, torch.where(any_valid, low, 0), torch.where(any_valid, high, 0), var]


def trajectory_features(z: torch.Tensor, valid: torch.Tensor, early_steps: int, early_angles: int) -> torch.Tensor:
    """(B, T, D) points and (B, T) mask to (B, early_steps + early_angles + 8) features.

    A step counts when both its frames are genuine, an angle when all three are;
    leading entries beyond a clip's genuine frames are zero.
    """
    d, theta = temporal_geometry(z)
    step_valid = valid[:, 1:] & valid[:, :-1]
    angle_valid = step_valid[:, 1:] & step_valid[:, :-1]
    d, theta = d * step_valid, theta * angle_valid
    stats = [*masked_moments(d, step_valid), *masked_moments(theta, angle_valid)]
    return torch.cat([d[:, :early_steps], theta[:, :early_angles], torch.stack(stats, dim=1)], dim=1)
