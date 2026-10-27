"""ViF-Bench fields read from configs/datasets/vifbench.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


_SKIP = {"dataset"}

_DEFAULT_GENERATORS = [
    "CogVideoX1.5-5B-T",
    "HunyuanVideo",
    "HunyuanVideo-I2V",
    "LTX-Video-13B-I",
    "LTX-Video-13B-T",
    "SkyReels-V2",
    "SkyReels-V2-I2V-14B-540P",
    "Wan2.1-T2V-1.3B",
    "Wan2.1-VACE-1.3B-T",
    "Wan2.2-I2V-14B",
    "Wan2.2-T2V-14B",
    "Wan2.2-TI2V-5B-I",
    "Wan2.2-TI2V-5B-T",
    "gen4-turbo",
    "hailuo",
    "kling-v1",
    "pika-v2",
    "pixverse-v4-5",
    "sora-2",
]


@dataclass
class VifbenchConfig:
    """On-disk layout for ViF-Bench source videos: Real mp4s and Fake per generator."""

    root: str
    roles: list[str] = field(default_factory=lambda: ["test"])
    real_dir: str = "Real"
    fake_dir: str = "Fake"
    generators: list[str] = field(default_factory=lambda: list(_DEFAULT_GENERATORS))
    manifest: str = "artifacts/vifbench/manifest.csv"
    max_videos_per_generator: int | None = None

    def __post_init__(self) -> None:
        """Require a test role and at least one generator."""
        if not self.roles:
            self.roles = ["test"]
        if "test" not in self.roles:
            raise ValueError(f"vifbench roles must include test, got {self.roles}")
        if not self.generators:
            raise ValueError("vifbench generators must not be empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VifbenchConfig:
        """Build from a corpus YAML mapping (plus optional --set overlays)."""
        if not data.get("root"):
            raise ValueError("vifbench requires root (set it in configs/datasets/vifbench.yaml)")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known - _SKIP
        if unknown:
            raise ValueError(f"unknown ViF-Bench config keys: {sorted(unknown)}")
        payload = {k: v for k, v in data.items() if k in known}
        return cls(**payload)
