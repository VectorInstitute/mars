"""DFD-HR (Sun et al., CVPR 2026), re-implemented under the shared protocol."""

from .config import Config, load_config
from .detector import DFDHRDetector
from .model import DFDHR, DFDHROutput, build_model


__all__ = ["DFDHR", "Config", "DFDHRDetector", "DFDHROutput", "build_model", "load_config"]
