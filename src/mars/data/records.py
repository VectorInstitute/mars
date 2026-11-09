"""One video row. Required fields plus optional extras per corpus."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VideoRecord:
    """One video on disk.

    Required: identity, path, binary label, frame count, split. Corpus-specific
    columns (FF++ ``method``, compression, group, ...) live in ``extra``.
    """

    video_id: str
    path: str
    label: int
    n_frames: int
    split: str
    extra: dict[str, Any] = field(default_factory=dict)
