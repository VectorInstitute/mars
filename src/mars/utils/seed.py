"""RNG seeding."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, rank: int = 0) -> None:
    """Seed every RNG. Ranks are offset so workers do not draw identical streams."""
    effective = seed + rank
    os.environ["PYTHONHASHSEED"] = str(effective)
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    torch.cuda.manual_seed_all(effective)
