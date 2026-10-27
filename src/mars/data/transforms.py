"""Shared clip preprocessing: resize and normalise RGB frames."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_MEAN = 0.5
DEFAULT_STD = 0.5


def preprocess(
    frames: np.ndarray,
    image_size: int = 384,
    mean: float = DEFAULT_MEAN,
    std: float = DEFAULT_STD,
) -> torch.Tensor:
    """``(N, H, W, 3)`` uint8 RGB -> ``(N, 3, S, S)`` float32.

    Default ``mean``/``std`` of 0.5 map pixels to ``[-1, 1]``. Pass other
    values for a different encoder; omitting them keeps current behaviour.
    """
    tensor = torch.from_numpy(np.ascontiguousarray(frames))
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    if tensor.shape[-2:] != (image_size, image_size):
        tensor = F.interpolate(
            tensor,
            size=(image_size, image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp_(0.0, 1.0)
    return tensor.sub_(mean).div_(std)
