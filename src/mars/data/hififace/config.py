"""HifiFace fields read from configs/datasets/hififace_*.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


_SKIP = {"dataset"}
_VARIANTS = {"paired", "onlyfake"}


@dataclass
class HififaceConfig:
    """On-disk layout for HifiFace: fakes under root, FF++ originals when paired."""

    root: str
    variant: str
    roles: list[str] = field(default_factory=lambda: ["test"])
    ffpp_root: str | None = None
    compression: str = "c23"
    originals: list[str] = field(default_factory=lambda: ["youtube"])
    split_file: str | None = None
    manifest: str = "artifacts/hififace/manifest.csv"
    max_videos_per_class: int | None = None

    def __post_init__(self) -> None:
        """Require a test role; HifiFace is not a train corpus."""
        if not self.roles:
            self.roles = ["test"]
        if "test" not in self.roles:
            raise ValueError(f"hififace roles must include test, got {self.roles}")
        if self.variant not in _VARIANTS:
            raise ValueError(f"hififace variant must be one of {sorted(_VARIANTS)}, got {self.variant!r}")
        if self.variant == "paired" and not self.ffpp_root:
            raise ValueError("hififace paired requires ffpp_root")
        if self.variant == "paired" and not self.originals:
            raise ValueError("hififace paired requires originals")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HififaceConfig:
        """Build from a corpus YAML mapping (plus optional --set overlays)."""
        if not data.get("root"):
            raise ValueError("hififace requires root (set it in configs/datasets/hififace_*.yaml)")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known - _SKIP
        if unknown:
            raise ValueError(f"unknown HifiFace config keys: {sorted(unknown)}")
        payload = {k: v for k, v in data.items() if k in known}
        return cls(**payload)
