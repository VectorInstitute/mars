"""Manifest file paths → vLLM chat conversation."""

from __future__ import annotations

from deepfake_moe.models.base import ClipData
from deepfake_moe.models.vllm_config import VllmConfig


def build_conversation(
    clip: ClipData,
    *,
    user_text: str,
    system_prompt: str,
    cfg: VllmConfig,
) -> list[dict]:
    """Build a chat message list for ``apply_chat_template`` / ``process_mm_info``."""
    content: list[dict] = []
    if clip.video_path:
        content.append(
            {
                "type": "video",
                "video": clip.video_path,
                "max_frames": cfg.video_max_frames,
                "min_frames": cfg.video_min_frames,
            }
        )
    if clip.has_audio and clip.audio_path:
        content.append({"type": "audio", "audio": clip.audio_path})
    content.append({"type": "text", "text": user_text})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
