"""MARS: multiscale appearance-residual streams for video forgery detection."""

from .config import Config, ModelConfig, StreamsConfig, load_config
from .detector import MarsDetector
from .model import Mars, build_model
from .residual_path import MarsOutput, ResidualPath


__all__ = [
    "Config",
    "Mars",
    "MarsDetector",
    "MarsOutput",
    "ResidualPath",
    "ModelConfig",
    "StreamsConfig",
    "build_model",
    "load_config",
]
