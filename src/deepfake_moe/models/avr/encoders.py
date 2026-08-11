"""Frozen audio/visual encoder loading for AVR. AVR never trains these encoders."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import Qwen3OmniMoeThinkerForConditionalGeneration


@dataclass
class EncoderOutputs:
    """Native-length token streams AVR operates on, before any masking/fusion.

    ``video_hidden`` is the encoder's ``pooler_output``, not
    ``last_hidden_state`` — ``pooler_output`` is the spatially-merged stream
    the Thinker actually scatters into video placeholder positions;
    ``last_hidden_state`` is the finer pre-merge patch stream the decoder
    never sees directly. AVR's evidence tokens need to land at the same
    per-token granularity as the real placeholder tokens.
    """

    audio_hidden: torch.Tensor  # (N_a, hidden_dim)
    video_hidden: torch.Tensor  # (N_v, hidden_dim)


def load_frozen_encoders(model_id: str) -> Qwen3OmniMoeThinkerForConditionalGeneration:
    """Load a Thinker checkpoint, keep only its frozen BF16 audio/visual towers.

    Loads ``Qwen3OmniMoeThinkerForConditionalGeneration`` directly rather than
    the outer composite model, since Talker/code2wav weights load as
    unexpected keys against the Thinker class and are dropped automatically.
    BF16, not NF4: nothing here is ever trained, so there's no reason to
    quantize. ``device_map="auto"`` shards the full checkpoint across every
    visible GPU during load (matching ``train/qlora_sft/model.py``'s
    precedent) — without it, ``from_pretrained`` puts the entire ~60GB BF16
    decoder on a single device before we ever reach the deletion below,
    which is enough to OOM a single GPU or the host's CPU RAM. ``model`` (the
    text decoder) and ``lm_head`` are deleted right after load to free their
    memory — ``get_audio_features``/``get_video_features`` only touch
    ``audio_tower``/``visual`` internally, so the trimmed instance still
    works for encoding.
    """
    thinker = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto"
    )
    del thinker.model
    del thinker.lm_head
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    thinker.audio_tower.eval()
    thinker.visual.eval()
    for module in (thinker.audio_tower, thinker.visual):
        for param in module.parameters():
            param.requires_grad = False
    return thinker


@torch.no_grad()
def encode(
    thinker: Qwen3OmniMoeThinkerForConditionalGeneration,
    *,
    input_features: torch.Tensor,
    feature_attention_mask: torch.Tensor,
    pixel_values_videos: torch.Tensor,
    video_grid_thw: torch.Tensor,
) -> EncoderOutputs:
    """Run the frozen encoders on one processor batch, returning native-length streams.

    Uses ``get_audio_features``/``get_video_features`` rather than calling
    ``audio_tower``/``visual`` directly, since those methods already handle
    the mask-to-length conversion (audio) and dtype cast (video) the raw
    towers require as separate preprocessing steps. ``get_video_features``
    casts ``pixel_values_videos`` to the tower's dtype internally, but
    ``get_audio_features`` does not do the same for ``input_features`` (it
    stays whatever dtype the processor produced, typically float32), so that
    cast happens here instead.
    """
    input_features = input_features.to(thinker.audio_tower.dtype)
    audio_hidden = thinker.get_audio_features(input_features, feature_attention_mask).last_hidden_state
    video_hidden = thinker.get_video_features(pixel_values_videos, video_grid_thw).pooler_output
    return EncoderOutputs(audio_hidden=audio_hidden, video_hidden=video_hidden)
