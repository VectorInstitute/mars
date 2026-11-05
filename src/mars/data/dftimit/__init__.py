"""DeepfakeTIMIT official test-set manifest and clip dataset."""

from mars.data.dftimit.dataset import DftimitDataset
from mars.data.dftimit.manifest import (
    ensure_manifest,
    load_records,
)
from mars.data.dftimit.manifest import (
    read as read_manifest,
)


__all__ = [
    "DftimitDataset",
    "ensure_manifest",
    "load_records",
    "read_manifest",
]
