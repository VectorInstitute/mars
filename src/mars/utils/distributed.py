"""Minimal DDP helpers. Works on a single GPU or CPU."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


# Ranks score uneven shards in evaluation (frame-level models skip padded frames),
# so a fast rank can wait long at the next gather; the default 10 minutes is too short.
COLLECTIVE_TIMEOUT = timedelta(hours=2)


@dataclass
class DistributedContext:
    """Process-group state for one rank."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device = torch.device("cpu")
    enabled: bool = False

    @property
    def is_main(self) -> bool:
        """True on rank 0."""
        return self.rank == 0

    def barrier(self) -> None:
        """Synchronize all ranks when DDP is enabled."""
        if self.enabled:
            dist.barrier()

    def cleanup(self) -> None:
        """Destroy the process group if it was created."""
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()

    def all_gather_object(self, obj: Any) -> list[Any]:
        """Gather a picklable object from every rank."""
        if not self.enabled:
            return [obj]
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, obj)
        return gathered

    def broadcast_(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """In-place broadcast of ``tensor`` from ``src``."""
        if self.enabled:
            dist.broadcast(tensor, src=src)
        return tensor


def setup_distributed() -> DistributedContext:
    """Initialize from torchrun environment variables when present."""
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) == 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return DistributedContext(device=device)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, timeout=COLLECTIVE_TIMEOUT)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return DistributedContext(rank, world_size, local_rank, device, enabled=True)
