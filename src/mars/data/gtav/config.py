"""GTA-V fields read from configs/datasets/gtav.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


_SKIP = {"dataset"}


@dataclass
class GtavConfig:
    """On-disk layout for SAIL-VOS-3D: one images/ folder per clip."""

    root: str
    roles: list[str] = field(default_factory=lambda: ["train", "test"])
    images_dir: str = "images"
    split_dir: str = "split"
    train_file: str = "split_train.txt"
    test_file: str = "split_val.txt"
    manifest: str = "artifacts/gtav/manifest.csv"
    max_videos_per_class: int | None = None

    def __post_init__(self) -> None:
        """Require train and test; GTA-V has no val split."""
        if not self.roles:
            self.roles = ["train", "test"]
        missing = [role for role in ("train", "test") if role not in self.roles]
        if missing:
            raise ValueError(f"gtav roles must include train and test, got {self.roles}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GtavConfig:
        """Build from a corpus YAML mapping (plus optional --set overlays)."""
        if not data.get("root"):
            raise ValueError("gtav requires root (set it in configs/datasets/gtav.yaml)")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known - _SKIP
        if unknown:
            raise ValueError(f"unknown GTA-V config keys: {sorted(unknown)}")
        payload = {k: v for k, v in data.items() if k in known}
        return cls(**payload)
