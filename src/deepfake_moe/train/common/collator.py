"""Qwen multimodal collation with assistant-response-only labels."""

from __future__ import annotations

from typing import Any

import torch

from deepfake_moe.eval.prompts import SYSTEM_PROMPT


_ASSISTANT_MARKER = "<|im_start|>assistant\n"


def _last_subsequence(sequence: list[int], subsequence: list[int]) -> int | None:
    if not subsequence:
        return None
    for index in range(len(sequence) - len(subsequence), -1, -1):
        if sequence[index : index + len(subsequence)] == subsequence:
            return index
    return None


class SFTCollator:
    """Build processor inputs and mask prompt, media, and padding tokens."""

    def __init__(self, processor: Any, *, use_audio_in_video: bool = False) -> None:
        self.processor = processor
        self.use_audio_in_video = use_audio_in_video
        self.tokenizer = processor.tokenizer
        self._assistant_ids = self.tokenizer.encode(_ASSISTANT_MARKER, add_special_tokens=False)

    def _prompt(self, feature: dict[str, Any]) -> str:
        clip = feature["clip"]
        processor = self.processor
        media = processor.vision_bos_token + processor.video_token + processor.vision_eos_token
        if clip.has_audio and clip.waveform is not None:
            media += processor.audio_bos_token + processor.audio_token + processor.audio_eos_token
        eos = self.tokenizer.eos_token
        user_turn = f"<|im_start|>user\n{media}{feature['user_prompt']}{eos}\n{_ASSISTANT_MARKER}"
        return f"<|im_start|>system\n{SYSTEM_PROMPT}{eos}\n{user_turn}"

    def _process(self, features: list[dict[str, Any]], texts: list[str]) -> dict[str, torch.Tensor]:
        videos = [feature["clip"].frames for feature in features]
        audios = [
            feature["clip"].waveform
            for feature in features
            if feature["clip"].has_audio and feature["clip"].waveform is not None
        ]
        batch = self.processor(
            text=texts,
            audio=audios or None,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )
        return dict(batch)

    def generation_inputs(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Build prompt-only inputs for generated validation."""
        if not features:
            raise ValueError("Cannot collate an empty batch")
        return self._process(features, [self._prompt(feature) for feature in features])

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate decoded examples and create response-only language-model labels."""
        if not features:
            raise ValueError("Cannot collate an empty batch")
        prompts = [self._prompt(feature) for feature in features]
        eos = self.tokenizer.eos_token
        texts = [prompt + feature["target_text"] + eos for prompt, feature in zip(prompts, features, strict=True)]
        batch = self._process(features, texts)
        labels = batch["input_ids"].clone()
        attention_mask = batch.get("attention_mask")
        for row_index in range(labels.shape[0]):
            token_ids = batch["input_ids"][row_index].tolist()
            marker_start = _last_subsequence(token_ids, self._assistant_ids)
            if marker_start is None:
                raise ValueError("Assistant boundary token sequence not found in processed training input")
            response_start = marker_start + len(self._assistant_ids)
            labels[row_index, :response_start] = -100
            if attention_mask is not None:
                labels[row_index, attention_mask[row_index] == 0] = -100
        batch["labels"] = labels
        return batch
