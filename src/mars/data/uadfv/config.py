"""UADFV fields read from configs/datasets/uadfv.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


_SKIP = {"dataset"}


@dataclass
class UadfvConfig:
    """On-disk layout for UADFV: two folders of mp4 files."""

    root: str
    roles: list[str] = field(default_factory=lambda: ["test"])
    real_dir: str = "real"
    fake_dir: str = "fake"
    manifest: str = "artifacts/uadfv/manifest.csv"
    max_videos_per_class: int | None = None

    def __post_init__(self) -> None:
        """Require a test role; UADFV is not a train corpus."""
        if not self.roles:
            self.roles = ["test"]
        if "test" not in self.roles:
            raise ValueError(f"uadfv roles must include test, got {self.roles}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UadfvConfig:
        """Build from a corpus YAML mapping (plus optional --set overlays)."""
        if not data.get("root"):
            raise ValueError("uadfv requires root (set it in configs/datasets/uadfv.yaml)")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known - _SKIP
        if unknown:
            raise ValueError(f"unknown UADFV config keys: {sorted(unknown)}")
        payload = {k: v for k, v in data.items() if k in known}
        return cls(**payload)
