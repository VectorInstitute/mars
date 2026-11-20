"""MARScope: see what a video deepfake detector sees."""

from importlib.metadata import PackageNotFoundError, version

from marscope.explain.policy import apply_policy
from marscope.pipeline import MARScope
from marscope.schema import PolicyNote, VideoResult


try:
    __version__ = version("marscope")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

__all__ = ["MARScope", "PolicyNote", "VideoResult", "__version__", "apply_policy"]
