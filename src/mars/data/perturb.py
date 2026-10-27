"""Test-time degradations of decoded frames, for the robustness evaluation.

Applied to uint8 frames at their native resolution, before any model's own resize and
normalization, so every detector sees the same degraded frames.
"""

from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


KINDS = ("none", "jpeg", "blur", "resize", "noise")
# Gaussian blur kernel size per sigma, as in GenD's robustness test.
BLUR_KERNELS = {0.5: 5, 1.0: 7, 1.5: 11, 2.0: 13, 3.0: 19}


def check(kind: str, level: float) -> None:
    """Raise ValueError if the perturbation kind or level is invalid."""
    if kind not in KINDS:
        raise ValueError(f"unknown perturbation {kind!r}; expected one of {KINDS}")
    if kind == "jpeg" and not 1 <= level <= 100:
        raise ValueError(f"JPEG quality must be in [1, 100], got {level}")
    if kind == "blur" and level not in BLUR_KERNELS:
        raise ValueError(f"blur sigma must be one of {sorted(BLUR_KERNELS)}, got {level}")
    if kind == "resize" and level < 1:
        raise ValueError(f"resize short side must be at least 1 pixel, got {level}")
    if kind == "noise" and level < 0:
        raise ValueError(f"noise std must be non-negative, got {level}")


def _jpeg(frames: np.ndarray, quality: int) -> np.ndarray:
    out = np.empty_like(frames)
    for i, frame in enumerate(frames):
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="JPEG", quality=quality)
        out[i] = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))
    return out


def _blur(frames: np.ndarray, sigma: float) -> np.ndarray:
    size = BLUR_KERNELS[sigma]
    offsets = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    kernel = torch.exp(-(offsets**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
    channels = x.shape[1]
    pad = size // 2
    x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, 1, size).repeat(channels, 1, 1, 1), groups=channels)
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, size, 1).repeat(channels, 1, 1, 1), groups=channels)
    return x.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()


def _resize(frames: np.ndarray, short_side: int) -> np.ndarray:
    h, w = frames.shape[1:3]
    scale = short_side / min(h, w)
    size = (max(1, round(h * scale)), max(1, round(w * scale)))
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=scale < 1)
    return x.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()


def _noise(frames: np.ndarray, std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = frames.astype(np.float32) / 255.0 + rng.normal(0.0, std, size=frames.shape).astype(np.float32)
    return (np.clip(noisy, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def perturb(frames: np.ndarray, kind: str, level: float, seed: int = 0) -> np.ndarray:
    """Degrade (N, H, W, 3) uint8 frames.

    Parameters
    ----------
    frames : np.ndarray
        Decoded frames at their native resolution.
    kind : str
        none, jpeg (quality), blur (sigma), resize (short side in pixels) or noise (std
        on pixels in [0, 1]).
    level : float
        Strength of the perturbation, in the unit of its kind.
    seed : int
        Seed of the noise; ignored by the other kinds.

    Returns
    -------
    np.ndarray
        Degraded uint8 frames; resize changes their height and width.
    """
    check(kind, level)
    if kind == "none":
        return frames
    if kind == "jpeg":
        return _jpeg(frames, int(level))
    if kind == "blur":
        return _blur(frames, float(level))
    if kind == "resize":
        return _resize(frames, int(level))
    return _noise(frames, float(level), seed)
