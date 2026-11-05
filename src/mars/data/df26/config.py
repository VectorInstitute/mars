"""DF26 fields read from configs/datasets/df26.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


_SKIP = {"dataset"}

_DEFAULT_GENERATORS = [
    "Grok_Imagine",
    "HunyuanVideo_1.5_14b_i2v",
    "HunyuanVideo_1.5_14b_t2v",
    "Kling_3.0",
    "LTX_2.3_distilled_i2v",
    "LTX_2.3_distilled_t2v",
    "Veo_3.1",
    "Wan_2.2_14b_i2v",
    "Wan_2.2_14b_t2v",
    "Wan_2.6",
]


@dataclass
class Df26Config:
    """DF26 layout: the release's own video list, and which generators to score."""

    root: str
    roles: list[str] = field(default_factory=lambda: ["test"])
    metadata: str = "metadata/videos.csv"
    generators: list[str] = field(default_factory=lambda: list(_DEFAULT_GENERATORS))
    manifest: str = "artifacts/df26/manifest.csv"

    def __post_init__(self) -> None:
        """Allow the test role only, since the licence forbids training on DF26."""
        if self.roles != ["test"]:
            raise ValueError(f"df26 is evaluation-only, roles must be [test], got {self.roles}")
        if not self.generators:
            raise ValueError("df26 generators must not be empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Df26Config:
        """Build from a corpus YAML mapping (plus optional --set overlays)."""
        if not data.get("root"):
            raise ValueError("df26 requires root (set it in configs/datasets/df26.yaml)")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known - _SKIP
        if unknown:
            raise ValueError(f"unknown DF26 config keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in data.items() if k in known})
