"""Dataset loaders. Shared clip I/O lives here; each corpus is a subpackage."""

from mars.data.base import BaseDataset, Sample
from mars.data.config import FAKE, REAL, DataConfig
from mars.data.dataset import ClipDataset, ConcatClipDataset, collate
from mars.data.factory import build_dataset, build_split_datasets, load_records
from mars.data.records import VideoRecord
from mars.data.sampling import Clip, clips_for_video, sampled_indices, segment
from mars.data.transforms import preprocess
from mars.data.video import count_frames, list_frame_files, read_frames


__all__ = [
    "BaseDataset",
    "Clip",
    "ClipDataset",
    "ConcatClipDataset",
    "DataConfig",
    "FAKE",
    "REAL",
    "Sample",
    "VideoRecord",
    "build_dataset",
    "build_split_datasets",
    "clips_for_video",
    "collate",
    "count_frames",
    "list_frame_files",
    "load_records",
    "preprocess",
    "read_frames",
    "sampled_indices",
    "segment",
]
