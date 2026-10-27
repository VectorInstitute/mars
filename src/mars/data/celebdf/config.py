"""Celeb-DF v2 fields read from ``configs/datasets/celebdf.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


_SKIP = {"dataset"}


@dataclass
class CelebdfConfig:
    """On-disk layout and official test list for Celeb-DF v2.

    ``list_file`` is relative to ``root`` unless it is an absolute path.
    Labels come from the source folder, not the official list column.
    """

    root: str
    roles: list[str] = field(default_factory=lambda: ["test"])
    list_file: str = "List_of_testing_videos.txt"
    manifest: str = "artifacts/celebdf/manifest.csv"
    max_videos_per_class: int | None = None

    def __post_init__(self) -> None:
        """Require a test role; Celeb-DF is not a train corpus."""
        if not self.roles:
            self.roles = ["test"]
        if "test" not in self.roles:
            raise ValueError(f"celebdf roles must include test, got {self.roles}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CelebdfConfig:
        """Build from a corpus YAML mapping (plus optional ``--set`` overlays)."""
        if not data.get("root"):
            raise ValueError("celebdf requires root (set it in configs/datasets/celebdf.yaml)")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known - _SKIP
        if unknown:
            raise ValueError(f"unknown Celeb-DF config keys: {sorted(unknown)}")
        payload = {k: v for k, v in data.items() if k in known}
        return cls(**payload)
