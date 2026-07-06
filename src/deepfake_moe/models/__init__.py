"""Model access."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from deepfake_moe.models.hf.qwen_omni import QwenOmni

__all__ = ["QwenOmni"]


def __getattr__(name: str) -> object:
    if name == "QwenOmni":
        from deepfake_moe.models.hf.qwen_omni import QwenOmni  # noqa: PLC0415

        return QwenOmni
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
