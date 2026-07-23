"""Qwen3-Omni: load, raw forward, generate, and eval ``predict``."""

from __future__ import annotations

from typing import Any, cast

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from deepfake_moe.eval.logprob import bool_token_ids, field_value_step_indices, stream_probs_from_step_logprobs
from deepfake_moe.eval.parser import parse_response
from deepfake_moe.eval.prompts import AV_USER_PROMPT, SYSTEM_PROMPT, VIDEO_ONLY_USER_PROMPT
from deepfake_moe.models.base import ClipData, PredictionOutput
from deepfake_moe.models.config import Config
from deepfake_moe.models.hf import device_routing
from deepfake_moe.utils.logging_utils import get_logger


logger = get_logger(__name__)


class QwenOmni:
    """Load Qwen3-Omni once; raw forward for inspection, chat for comparison.

    Also implements the ``Predictor`` protocol via ``predict()`` for the eval harness.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model: Any = None
        self.processor: Any = None
        self._true_token_ids: set[int] = set()
        self._false_token_ids: set[int] = set()

    def load(self) -> "QwenOmni":
        """Load model + processor; returns self for chaining."""
        cfg = self.config
        kwargs: dict[str, Any] = {"dtype": cfg.dtype, "device_map": cfg.device_map}
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = cfg.attn_implementation

        logger.info("Loading Qwen3-Omni: %s", cfg.model_id)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(cfg.model_id, **kwargs)
        self.model.eval()
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(cfg.model_id)
        self._true_token_ids, self._false_token_ids = bool_token_ids(self.processor.tokenizer)
        if not self._true_token_ids or not self._false_token_ids:
            logger.warning("Tokenizer has no single-token JSON true/false pair; logprob capture disabled.")
        return self

    def predict(self, clip: ClipData) -> PredictionOutput:
        """Run deepfake detection on one clip; implements the ``Predictor`` protocol.

        Uses the same in-memory media path as ``build_raw_inputs`` / ``run_forward``
        (not ``process_mm_info``, which expects file paths or PIL frames).
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        has_audio = clip.has_audio and clip.waveform is not None
        user_text = AV_USER_PROMPT if clip.has_audio else VIDEO_ONLY_USER_PROMPT
        audios = [clip.waveform] if has_audio else None
        videos = [clip.frames]

        inputs = self.build_raw_inputs(
            user_text,
            audio=audios,
            videos=videos,
            system_prompt=SYSTEM_PROMPT,
        )
        generated = self._generate_from_inputs(inputs, capture_logprobs=True)
        raw, token_ids, step_logprobs = cast(tuple[str, list[int], list[dict[int, float]]], generated)
        prediction = parse_response(raw, has_audio=clip.has_audio, clip_id=clip.clip_id)
        steps = field_value_step_indices(
            token_ids,
            self.processor.tokenizer,
            self._true_token_ids,
            self._false_token_ids,
        )
        video_step = steps.get("video_fake")
        if video_step is not None:
            video_prob = stream_probs_from_step_logprobs(
                step_logprobs[video_step],
                self._true_token_ids,
                self._false_token_ids,
            )
            if video_prob is not None:
                prediction.video_fake_prob = video_prob
        audio_step = steps.get("audio_fake")
        if clip.has_audio and audio_step is not None:
            audio_prob = stream_probs_from_step_logprobs(
                step_logprobs[audio_step],
                self._true_token_ids,
                self._false_token_ids,
            )
            if audio_prob is not None:
                prediction.audio_fake_prob = audio_prob
        return prediction

    def _eval_prompt(
        self,
        user_text: str,
        *,
        has_audio: bool,
        has_video: bool,
        system_prompt: str | None = None,
    ) -> str:
        """System + user prompt with media placeholder tokens (raw forward style)."""
        user_turn = self._prompt(user_text, audio=has_audio, video=has_video, image=False)
        if not system_prompt:
            return user_turn
        eos = self.processor.tokenizer.eos_token
        return f"<|im_start|>system\n{system_prompt}{eos}\n{user_turn}"

    @property
    def thinker(self) -> Any:
        """Understanding submodule (``audio_tower``, ``visual``, ``model``)."""
        return getattr(self.model, "thinker", self.model)

    def _prompt(self, text: str, *, audio: bool, video: bool, image: bool) -> str:
        """Build a user turn using tokenizer special tokens."""
        proc = self.processor
        media = ""
        if video:
            media += proc.vision_bos_token + proc.video_token + proc.vision_eos_token
        if image:
            media += proc.vision_bos_token + proc.image_token + proc.vision_eos_token
        if audio:
            media += proc.audio_bos_token + proc.audio_token + proc.audio_eos_token
        end = proc.tokenizer.eos_token
        return f"<|im_start|>user\n{media}{text}{end}\n<|im_start|>assistant\n"

    def _build_conversation(
        self,
        text: str,
        *,
        video: Any | None = None,
        audio: Any | None = None,
        images: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Chat message list for ``apply_chat_template`` / ``process_mm_info``."""
        content: list[dict[str, Any]] = []
        if video is not None:
            if hasattr(video, "shape") and getattr(video, "ndim", 0) == 4:
                content.append({"type": "video", "video": list(video)})
            else:
                content.append({"type": "video", "video": video})
        if images is not None:
            content.append({"type": "image", "image": images})
        if audio is not None:
            content.append({"type": "audio", "audio": audio})
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def build_raw_inputs(
        self,
        text: str,
        *,
        audio: Any | None = None,
        images: Any | None = None,
        videos: Any | None = None,
        system_prompt: str | None = None,
    ) -> Any:
        """Build inputs without ``apply_chat_template``."""
        has_audio = audio is not None
        prompt = self._eval_prompt(
            text,
            has_audio=has_audio,
            has_video=videos is not None,
            system_prompt=system_prompt,
        )
        return self.processor(
            text=[prompt],
            audio=audio,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.config.use_audio_in_video,
        )

    # Device routing (multi-GPU device_map placement) lives in device_routing.py,
    # shared with train/qlora_sft/trainer.py's SFTTrainer. These stay as thin
    # delegating wrappers so the public QwenOmni surface (and existing tests)
    # don't need to change.
    @staticmethod
    def _parse_hf_device(dev: Any) -> torch.device:
        return device_routing.parse_hf_device(dev)

    @staticmethod
    def _module_device(module: Any) -> torch.device:
        return device_routing.module_device(module)

    def _thinker_input_devices(self, module: Any) -> dict[str, torch.device]:
        """Resolve per-modality devices when the model is sharded via ``device_map``."""
        return device_routing.thinker_input_devices(self.model, module)

    def _is_sharded_hf_model(self, module: Any) -> bool:
        """Return True when Accelerate dispatched submodules across multiple devices."""
        return device_routing.is_sharded_hf_model(self.model, module)

    @staticmethod
    def _tensor_device(key: str, devices: dict[str, torch.device]) -> torch.device:
        return device_routing.tensor_device(key, devices)

    def _prepare_inputs(self, inputs: Any, module: Any | None = None) -> Any:
        """Move tensors to the right devices for sharded Qwen3-Omni modules."""
        return device_routing.prepare_inputs(self.model, inputs, module=module)

    def raw_forward(
        self,
        inputs: Any,
        *,
        output_hidden_states: bool = True,
        output_router_logits: bool = False,
    ) -> Any:
        """Forward through the thinker; returns logits and hidden states."""
        inputs = self._prepare_inputs(inputs, self.thinker)
        with torch.inference_mode():
            return self.thinker(
                **inputs,
                output_hidden_states=output_hidden_states,
                output_router_logits=output_router_logits,
                return_dict=True,
            )

    def encoder_hooks(self) -> tuple[dict[str, Any], list[Any]]:
        """Register hooks on audio/visual encoders."""
        captured: dict[str, Any] = {}
        handles: list[Any] = []
        for key, attr in (("audio", "audio_tower"), ("visual", "visual")):
            module = getattr(self.thinker, attr, None)
            if module is None:
                logger.warning("Encoder '%s' not found; skipping hook.", attr)
                continue
            handles.append(module.register_forward_hook(lambda _m, _i, out, k=key: captured.__setitem__(k, out)))
        return captured, handles

    def router_hooks(self) -> tuple[dict[int, Any], list[Any]]:
        """Register hooks on MoE gate modules."""
        captured: dict[int, Any] = {}
        handles: list[Any] = []
        layers = getattr(self.thinker.model, "layers", None)
        if layers is None:
            logger.warning("Thinker decoder layers not found; skipping router hooks.")
            return captured, handles
        for layer_idx, layer in enumerate(layers):
            gate = getattr(layer.mlp, "gate", None)
            if gate is None:
                continue
            handles.append(
                gate.register_forward_hook(lambda _m, _i, out, idx=layer_idx: captured.__setitem__(idx, out))
            )
        return captured, handles

    def _generate_from_inputs(
        self,
        inputs: Any,
        *,
        max_new_tokens: int | None = None,
        capture_logprobs: bool = False,
    ) -> str | tuple[str, list[int], list[dict[int, float]]]:
        """Decode generated tokens from prepared processor inputs."""
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        use_av = self.config.use_audio_in_video
        inputs = self._prepare_inputs(inputs)

        generate_kwargs: dict[str, Any] = {}
        if capture_logprobs:
            generate_kwargs.update(return_dict_in_generate=True, output_scores=True)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_audio=False,
                use_audio_in_video=use_av,
                **generate_kwargs,
            )
        if hasattr(out, "sequences"):
            ids = out.sequences
        elif isinstance(out, (tuple, list)):
            ids = out[0]
        else:
            ids = out
        ids = ids[:, inputs.input_ids.shape[1] :]
        raw = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        if not capture_logprobs:
            return raw

        wanted_ids = self._true_token_ids | self._false_token_ids
        step_logprobs: list[dict[int, float]] = []
        for scores in out.scores:
            logprobs = torch.log_softmax(scores[0], dim=-1)
            step_logprobs.append({token_id: float(logprobs[token_id].item()) for token_id in wanted_ids})
        return raw, ids[0].tolist(), step_logprobs

    def chat_generate(self, conversation: list[dict[str, Any]], *, max_new_tokens: int | None = None) -> str:
        """Generate decoded text via chat template (no speech)."""
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        use_av = self.config.use_audio_in_video
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_av)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_av,
        )
        return self._generate_from_inputs(inputs, max_new_tokens=max_new_tokens)

    def generate(
        self,
        text: str,
        *,
        video: Any | None = None,
        audio: Any | None = None,
        images: Any | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate text from paths or in-memory media."""
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        conversation = self._build_conversation(text, video=video, audio=audio, images=images)
        return self.chat_generate(conversation, max_new_tokens=max_new_tokens)
