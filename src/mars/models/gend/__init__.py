"""GenD (Yermakov et al., WACV 2026), re-implemented under the shared protocol."""

from .config import Config, load_config
from .detector import GenDDetector
from .model import GenD, GenDOutput, build_model


__all__ = ["Config", "GenD", "GenDDetector", "GenDOutput", "build_model", "load_config"]
