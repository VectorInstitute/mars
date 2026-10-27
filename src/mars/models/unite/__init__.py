"""Unite detector: frozen SigLIP encoder, temporal transformer, CE + AD loss."""

from .config import Config, load_config
from .detector import UniteDetector
from .model import UNITE, build_model


__all__ = ["Config", "UNITE", "UniteDetector", "build_model", "load_config"]
