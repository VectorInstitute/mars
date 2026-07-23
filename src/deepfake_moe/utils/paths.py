"""Resolve manifest media paths onto the shared data root.

Manifest ``src_path`` values are relative to a per-dataset video root under
``DATA_ROOT``. ``wav_path`` may still carry a stale absolute prefix; anything
containing ``processed/audio/normalized/<rel>`` is rewritten onto ``AUDIO_DIR``.
"""

from __future__ import annotations

import re
from pathlib import Path


_AUDIO_NORMALIZED_RE = re.compile(r"(?:^|/)(?:processed/)?audio/normalized/(?P<rel>.+)$")


def default_audio_dir(data_root: str) -> str:
    """Return ``{data_root}/processed/audio/normalized``."""
    if not data_root:
        return ""
    return str(Path(data_root) / "processed" / "audio" / "normalized")


def resolve_video_path(
    src_path: str,
    *,
    data_root: str,
    video_root: str,
) -> str:
    """Join a relative ``src_path`` under ``data_root/video_root``.

    Absolute ``src_path`` values are returned unchanged (legacy / direct paths).
    """
    if not src_path:
        raise ValueError("src_path is empty")
    path = Path(src_path)
    if path.is_absolute():
        return str(path)
    if not data_root:
        raise ValueError("data_root is required to resolve a relative src_path")
    if not video_root:
        raise ValueError("video_root is required to resolve a relative src_path")
    return str(Path(data_root) / video_root / src_path)


def resolve_audio_path(wav_path: str | None, *, audio_dir: str) -> str | None:
    """Map a manifest ``wav_path`` onto ``audio_dir``.

    * Paths containing ``[processed/]audio/normalized/<rel>`` →
      ``audio_dir/<rel>`` (handles stale absolute paths and manifest-relative
      paths such as ``audio/normalized/lavdf/train/017033.wav``).
    * Relative paths → ``audio_dir/<wav_path>``.
    * Other absolute paths → unchanged.
    * Empty / ``None`` → ``None``.
    """
    if not wav_path:
        return None
    normalized = wav_path.replace("\\", "/")
    match = _AUDIO_NORMALIZED_RE.search(normalized)
    if match is not None:
        if not audio_dir:
            raise ValueError("audio_dir is required to remap normalized wav_path")
        return str(Path(audio_dir) / match.group("rel"))
    path = Path(wav_path)
    if path.is_absolute():
        return str(path)
    if not audio_dir:
        raise ValueError("audio_dir is required to resolve a relative wav_path")
    return str(Path(audio_dir) / wav_path)
