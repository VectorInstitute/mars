"""Celeb-DF v2 official test-list manifest and clip dataset."""

from mars.data.celebdf.dataset import CelebdfDataset
from mars.data.celebdf.manifest import (
    ensure_manifest,
    load_records,
)
from mars.data.celebdf.manifest import (
    read as read_manifest,
)


__all__ = [
    "CelebdfDataset",
    "ensure_manifest",
    "load_records",
    "read_manifest",
]
