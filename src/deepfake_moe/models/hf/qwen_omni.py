"""Qwen3-Omni: load, raw forward, generate, and eval ``predict``."""

from __future__ import annotations

from typing import Any

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from deepfake_moe.models.config import Config
from deepfake_moe.eval.parser import parse_response
from deepfake_moe.eval.prompts import AV_USER_PROMPT, SYSTEM_PROMPT, VIDEO_ONLY_USER_PROMPT
from deepfake_moe.utils.logging_utils import get_logger
from deepfake_moe.models.base import ClipData, PredictionOutput


logger = get_logger(__name__)


class QwenOmni:
    """Load Qwen3-Omni once; raw forward for inspection, chat for comparison.

    Also implements the ``Predictor`` protocol via ``predict()`` for the eval harness.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model: Any = None
        self.processor: Any = None
        self._logged_sharded_input_routing = False

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
        raw = self._generate_from_inputs(inputs)
        return parse_response(raw, has_audio=clip.has_audio, clip_id=clip.clip_id)

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

    @staticmethod
    def _parse_hf_device(dev: Any) -> torch.device:
        if isinstance(dev, torch.device):
            return dev
        if isinstance(dev, int):
            return torch.device(f"cuda:{dev}" if dev >= 0 else "cpu")
        if isinstance(dev, str):
            return torch.device(dev)
        if isinstance(dev, (list, tuple)) and dev:
            return QwenOmni._parse_hf_device(dev[0])
        raise TypeError(f"Unsupported device entry in hf_device_map: {dev!r}")

    @staticmethod
    def _module_device(module: Any) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _device_from_hf_map(self, hf_map: dict[str, Any], *name_fragments: str) -> torch.device | None:
        for name, dev in hf_map.items():
            if any(fragment in name for fragment in name_fragments):
                return self._parse_hf_device(dev)
        return None

    def _thinker_input_devices(self, module: Any) -> dict[str, torch.device]:
        """Resolve per-modality devices when the model is sharded via ``device_map``."""
        root = self.model if self.model is not None else module
        thinker = getattr(root, "thinker", module)
        hf_map = getattr(root, "hf_device_map", None) or getattr(module, "hf_device_map", None) or {}

        embed = getattr(getattr(thinker, "model", None), "embed_tokens", None)
        default = (
            self._device_from_hf_map(hf_map, "thinker.model.embed_tokens", "model.embed_tokens", "embed_tokens")
            or (self._module_device(embed) if embed is not None else None)
            or self._module_device(thinker)
        )
        visual_mod = getattr(thinker, "visual", None)
        audio_mod = getattr(thinker, "audio_tower", None)
        visual = self._device_from_hf_map(hf_map, "thinker.visual", ".visual") or (
            self._module_device(visual_mod) if visual_mod is not None else default
        )
        audio = self._device_from_hf_map(hf_map, "thinker.audio_tower", "audio_tower") or (
            self._module_device(audio_mod) if audio_mod is not None else default
        )
        return {"default": default, "visual": visual, "audio": audio}

    def _is_sharded_hf_model(self, module: Any) -> bool:
        """Return True when Accelerate dispatched submodules across multiple devices."""
        root = self.model if self.model is not None else module
        hf_map = getattr(root, "hf_device_map", None) or getattr(module, "hf_device_map", None) or {}
        if not hf_map:
            return False
        devices = {str(self._parse_hf_device(dev)) for dev in hf_map.values()}
        return len(devices) > 1

    @staticmethod
    def _tensor_device(key: str, devices: dict[str, torch.device]) -> torch.device:
        key_lower = key.lower()
        if key_lower in {"input_features", "feature_attention_mask", "audio_feature_lengths"} or "audio" in key_lower:
            return devices["audio"]
        if any(token in key_lower for token in ("video", "pixel", "image", "grid_thw")):
            return devices["visual"]
        return devices["default"]

    def _prepare_inputs(self, inputs: Any, module: Any | None = None) -> Any:
        """Move tensors to the right devices for sharded Qwen3-Omni modules."""
        module = module or self.model
        devices = self._thinker_input_devices(module)
        dtype = getattr(module, "dtype", torch.float16)
        # For HF model-parallel dispatch (device_map with >1 device), multimodal tensors
        # must enter on one anchor device; splitting input tensors across modalities can
        # trigger internal concat/ops with mixed devices (cuda:0 vs cuda:1).
        if self._is_sharded_hf_model(module):
            target = devices["default"]
            if not self._logged_sharded_input_routing:
                logger.info("HF sharded device_map detected; routing all model inputs to %s", target)
                self._logged_sharded_input_routing = True
            for key in list(inputs.keys()):
                value = inputs[key]
                if not isinstance(value, torch.Tensor):
                    continue
                if value.is_floating_point():
                    inputs[key] = value.to(device=target, dtype=dtype)
                else:
                    inputs[key] = value.to(device=target)
            return inputs

        for key in list(inputs.keys()):
            value = inputs[key]
            if not isinstance(value, torch.Tensor):
                continue
            dev = self._tensor_device(key, devices)
            if value.is_floating_point():
                inputs[key] = value.to(device=dev, dtype=dtype)
            else:
                inputs[key] = value.to(device=dev)
        return inputs

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

    def _generate_from_inputs(self, inputs: Any, *, max_new_tokens: int | None = None) -> str:
        """Decode generated tokens from prepared processor inputs."""
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        use_av = self.config.use_audio_in_video
        inputs = self._prepare_inputs(inputs)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_audio=False,
                use_audio_in_video=use_av,
            )
        if hasattr(out, "sequences"):
            ids = out.sequences
        elif isinstance(out, (tuple, list)):
            ids = out[0]
        else:
            ids = out
        ids = ids[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]

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
