"""AVID fields read from configs/datasets/avid.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


_SKIP = {"dataset"}


@dataclass
class AvidConfig:
    """On-disk layout for AVID: two folders of mp4 files."""

    root: str
    roles: list[str] = field(default_factory=lambda: ["test"])
    real_dir: str = "real"
    fake_dir: str = "fake"
    manifest: str = "artifacts/avid/manifest.csv"
    max_videos_per_class: int | None = None

    def __post_init__(self) -> None:
        """Require a test role; AVID is not a train corpus."""
        if not self.roles:
            self.roles = ["test"]
        if "test" not in self.roles:
            raise ValueError(f"avid roles must include test, got {self.roles}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AvidConfig:
        """Build from a corpus YAML mapping (plus optional --set overlays)."""
        if not data.get("root"):
            raise ValueError("avid requires root (set it in configs/datasets/avid.yaml)")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known - _SKIP
        if unknown:
            raise ValueError(f"unknown AVID config keys: {sorted(unknown)}")
        payload = {k: v for k, v in data.items() if k in known}
        return cls(**payload)
