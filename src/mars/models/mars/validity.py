"""Which frames of a padded clip are genuine observations.

Clips are padded by repeating the last real frame up to a fixed length. The mask
is a contiguous prefix of ones, and every temporal operation is restricted to it.
"""

from __future__ import annotations

import torch


def is_prefix(valid: torch.Tensor) -> bool:
    """Report whether every row of (B, T) is ones followed only by zeros."""
    lengths = valid.sum(dim=1)
    reference = torch.arange(valid.shape[1], device=valid.device)[None, :] < lengths[:, None]
    return bool((reference == valid).all())


def observation_index(valid: torch.Tensor) -> torch.Tensor:
    """Position of each frame counted over genuine observations only, (B, T).

    Padded frames reuse the last genuine index, which keeps the lookup inside the
    encoding table; their outputs are discarded anyway.
    """
    return (valid.long().cumsum(dim=1) - 1).clamp(min=0)
