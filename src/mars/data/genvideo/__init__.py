"""GenVideo test-set manifest and clip dataset."""

from mars.data.genvideo.dataset import GenvideoDataset
from mars.data.genvideo.manifest import (
    ensure_manifest,
    load_records,
)
from mars.data.genvideo.manifest import (
    read as read_manifest,
)


__all__ = [
    "GenvideoDataset",
    "ensure_manifest",
    "load_records",
    "read_manifest",
]
