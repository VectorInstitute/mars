"""FaceForensics++ identity-safe manifest and clip dataset."""

from mars.data.ffpp.dataset import FfppDataset
from mars.data.ffpp.manifest import (
    ensure_manifest,
    load_records,
)
from mars.data.ffpp.manifest import (
    read as read_manifest,
)


__all__ = [
    "FfppDataset",
    "ensure_manifest",
    "load_records",
    "read_manifest",
]
