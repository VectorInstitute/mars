"""Shared clip sampling and which corpora this run trains/tests on."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from mars.data.perturb import check as check_perturbation


REAL, FAKE = 0, 1


def _as_str_list(value: Any) -> list[str]:
    """Accept a YAML string or list of strings."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


@dataclass
class DataConfig:
    """Clip sampling plus this run's train/val/test corpus names.

    Corpus identity (root, manifest, roles) stays in
    ``configs/datasets/<name>.yaml``. Unknown ``data:`` keys (for example
    ``manifest`` from ``--set``) overlay that YAML at load time.
    """

    train: list[str] = field(default_factory=list)
    val: str | None = None
    test: list[str] = field(default_factory=list)
    frame_stride: int = 2
    n_frames: int = 64
    clip_stride: int | None = None
    padding: str = "last_frame"
    image_size: int = 384
    num_workers: int = 8
    decode_backend: str = "pyav"
    # Test-time degradation of decoded frames (robustness evaluation); none is off.
    perturb: str = "none"
    perturb_level: float = 0.0
    overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize train / test / val after dataclass construction."""
        self.train = _as_str_list(self.train)
        self.test = _as_str_list(self.test)
        if not self.test and self.train:
            self.test = list(self.train)
        if not self.val:
            self.val = self.train[0] if self.train else None
        check_perturbation(self.perturb, self.perturb_level)

    @property
    def val_corpus(self) -> str:
        """Corpus used for checkpointing; defaults to the first train corpus."""
        name = self.val or (self.train[0] if self.train else "")
        if not name:
            raise ValueError("data.train is required")
        return name

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataConfig:
        """Build from a ``data:`` mapping.

        Sampling and ``train`` / ``val`` / ``test`` are typed fields. Any
        other key is kept as a corpus-YAML overlay (``overrides``).
        """
        if "dataset" in data:
            raise ValueError("data.dataset was replaced by data.train and data.test")
        known = {f.name for f in fields(cls)} - {"overrides"}
        overrides = dict(data.get("overrides") or {})
        overrides.update({k: v for k, v in data.items() if k not in known and k != "overrides"})
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs, overrides=overrides)
