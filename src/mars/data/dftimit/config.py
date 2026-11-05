"""DeepfakeTIMIT fields read from configs/datasets/dftimit.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


_SKIP = {"dataset"}


@dataclass
class DftimitConfig:
    """On-disk layout for DeepfakeTIMIT.

    Real videos are sentence directories of numbered JPEGs. Fakes are AVI
    files under each quality folder. Unused VidTIMIT subjects are ignored
    because they live outside real_dir.
    """

    root: str
    roles: list[str] = field(default_factory=lambda: ["test"])
    real_dir: str = "VidTIMIT-Real"
    fake_dir: str = "DeepfakeTIMIT-Fake"
    qualities: list[str] = field(default_factory=lambda: ["higher_quality", "lower_quality"])
    manifest: str = "artifacts/dftimit/manifest.csv"
    max_videos_per_class: int | None = None

    def __post_init__(self) -> None:
        """Require a test role; DeepfakeTIMIT is not a train corpus."""
        if not self.roles:
            self.roles = ["test"]
        if "test" not in self.roles:
            raise ValueError(f"dftimit roles must include test, got {self.roles}")
        if not self.qualities:
            raise ValueError("dftimit qualities must not be empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DftimitConfig:
        """Build from a corpus YAML mapping (plus optional --set overlays)."""
        if not data.get("root"):
            raise ValueError("dftimit requires root (set it in configs/datasets/dftimit.yaml)")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known - _SKIP
        if unknown:
            raise ValueError(f"unknown DeepfakeTIMIT config keys: {sorted(unknown)}")
        payload = {k: v for k, v in data.items() if k in known}
        return cls(**payload)
