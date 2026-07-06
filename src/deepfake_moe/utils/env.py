"""Load the repo-root ``.env`` and expand ``${VAR}`` references in config values.

Machine-specific paths (manifest/raw-data locations, FFmpeg prefix, etc.) live in
a gitignored ``.env`` at the repo root, so the committed YAML configs can stay
generic and reference them as ``${VAR}``. YAML has no native env support, so the
expansion happens here when a config is loaded.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"

_loaded = False


def load_env() -> None:
    """Load ``<repo>/.env`` into ``os.environ`` once (real env vars win)."""
    global _loaded  # noqa: PLW0603
    if not _loaded:
        load_dotenv(ENV_PATH, override=False)
        _loaded = True


def expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``$VAR`` in strings within dicts and lists.

    Loads ``.env`` on first use. Unset variables are left untouched so a missing
    ``.env`` produces a clear ``${VAR}``-shaped path rather than an empty string.

    Examples
    --------
    >>> import os
    >>> os.environ["DEMO_ROOT"] = "/data"
    >>> expand_env({"dir": "${DEMO_ROOT}/manifests"})
    {'dir': '/data/manifests'}
    """
    load_env()
    return _expand(value)


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value
