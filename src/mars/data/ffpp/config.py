"""FaceForensics++ fields read from ``configs/datasets/ffpp.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


_SKIP = {"dataset"}


@dataclass
class FfppConfig:
    """On-disk layout, roles, and identity-safe split settings for FF++.

    ``split_file`` is a directory of official ``train.json`` / ``val.json`` /
    ``test.json``. ``None`` uses ``src/mars/data/ffpp/splits`` when it holds train.json,
    otherwise ``split_fractions`` / ``split_seed``.
    """

    root: str
    roles: list[str] = field(default_factory=lambda: ["train", "val", "test"])
    manifest: str = "artifacts/ffpp/manifest.csv"
    compression: str = "c23"
    originals: list[str] = field(default_factory=lambda: ["youtube"])
    manipulations: list[str] = field(
        default_factory=lambda: [
            "Deepfakes",
            "Face2Face",
            "FaceSwap",
            "NeuralTextures",
        ]
    )
    split_fractions: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    split_seed: int = 1337
    split_file: str | None = None
    max_videos_per_class: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FfppConfig:
        """Build from a corpus YAML mapping (plus optional ``--set`` overlays)."""
        if not data.get("root"):
            raise ValueError("ffpp requires root (set it in configs/datasets/ffpp.yaml)")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known - _SKIP
        if unknown:
            raise ValueError(f"unknown FF++ config keys: {sorted(unknown)}")
        payload = {k: v for k, v in data.items() if k in known}
        ffpp = cls(**payload)
        if abs(sum(ffpp.split_fractions) - 1.0) > 1e-6:
            raise ValueError(f"split_fractions must sum to 1, got {ffpp.split_fractions}")
        return ffpp
