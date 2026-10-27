"""GenVideo fields read from configs/datasets/genvideo.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


_SKIP = {"dataset"}

_DEFAULT_GENERATORS = [
    "Crafter",
    "Gen2",
    "HotShot",
    "Lavie",
    "ModelScope",
    "MoonValley",
    "MorphStudio",
    "Show_1",
    "Sora",
    "WildScrape",
]


@dataclass
class GenvideoConfig:
    """On-disk layout for GenVideo-Val: Real mp4s and Fake per generator."""

    root: str
    roles: list[str] = field(default_factory=lambda: ["test"])
    real_dir: str = "Real"
    fake_dir: str = "Fake"
    generators: list[str] = field(default_factory=lambda: list(_DEFAULT_GENERATORS))
    manifest: str = "artifacts/genvideo/manifest.csv"
    max_videos_per_generator: int | None = None

    def __post_init__(self) -> None:
        """Require a test role and at least one generator."""
        if not self.roles:
            self.roles = ["test"]
        if "test" not in self.roles:
            raise ValueError(f"genvideo roles must include test, got {self.roles}")
        if not self.generators:
            raise ValueError("genvideo generators must not be empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenvideoConfig:
        """Build from a corpus YAML mapping (plus optional --set overlays)."""
        if not data.get("root"):
            raise ValueError("genvideo requires root (set it in configs/datasets/genvideo.yaml)")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known - _SKIP
        if unknown:
            raise ValueError(f"unknown GenVideo config keys: {sorted(unknown)}")
        payload = {k: v for k, v in data.items() if k in known}
        return cls(**payload)
