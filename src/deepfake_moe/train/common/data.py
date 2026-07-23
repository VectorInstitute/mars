"""Manifest-backed multimodal SFT dataset."""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from deepfake_moe.eval.dataset import load_clip
from deepfake_moe.eval.prompts import AV_USER_PROMPT, VIDEO_ONLY_USER_PROMPT
from deepfake_moe.train.common.targets import build_target, target_json


class SFTDataset(Dataset):
    """Decode manifest entries through the same media path used by evaluation."""

    def __init__(
        self,
        entries: list[dict],
        *,
        num_frames: int,
        audio_sample_rate: int,
        data_root: str = "",
        audio_dir: str = "",
        dataset_video_roots: dict[str, str] | None = None,
    ) -> None:
        self.entries = entries
        self.num_frames = num_frames
        self.audio_sample_rate = audio_sample_rate
        self.data_root = data_root
        self.audio_dir = audio_dir
        self.dataset_video_roots = dataset_video_roots

    def __len__(self) -> int:
        """Return the configured manifest-entry count."""
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Decode one entry and attach its prompt and target."""
        entry = self.entries[index]
        clip = load_clip(
            entry,
            num_frames=self.num_frames,
            sample_rate=self.audio_sample_rate,
            data_root=self.data_root,
            audio_dir=self.audio_dir,
            dataset_video_roots=self.dataset_video_roots,
        )
        if clip is None:
            clip_id = entry.get("clip_id", index)
            raise RuntimeError(f"Failed to decode training clip {clip_id!r}")
        return {
            "clip": clip,
            "user_prompt": AV_USER_PROMPT if clip.has_audio else VIDEO_ONLY_USER_PROMPT,
            "target": build_target(clip),
            "target_text": target_json(clip),
        }
