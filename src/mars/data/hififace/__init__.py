"""HifiFace test-set manifests and clip datasets."""

from mars.data.hififace.dataset import (
    HififaceOnlyfakeDataset,
    HififacePairedDataset,
)
from mars.data.hififace.manifest import (
    ensure_manifest,
    load_records,
)
from mars.data.hififace.manifest import (
    read as read_manifest,
)


__all__ = [
    "HififaceOnlyfakeDataset",
    "HififacePairedDataset",
    "ensure_manifest",
    "load_records",
    "read_manifest",
]
