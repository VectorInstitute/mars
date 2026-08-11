"""vLLM-backed Qwen3-Omni for Phase 1 eval (tensor parallel, file-path media)."""

from __future__ import annotations

import os
from importlib import metadata
from typing import Any

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor


try:
    from vllm import LLM, SamplingParams
except ModuleNotFoundError:
    LLM = None  # type: ignore[assignment, misc]
    SamplingParams = None  # type: ignore[assignment, misc]

from deepfake_moe.eval.logprob import bool_token_ids, field_value_step_indices, stream_probs_from_step_logprobs
from deepfake_moe.eval.parser import parse_response
from deepfake_moe.eval.prompts import AV_USER_PROMPT, SYSTEM_PROMPT, VIDEO_ONLY_USER_PROMPT
from deepfake_moe.models.base import ClipData, PredictionOutput
from deepfake_moe.models.vllm.media import build_conversation
from deepfake_moe.models.vllm_config import VllmConfig
from deepfake_moe.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _patch_qwen3_processor_for_vllm() -> bool:
    """Add compatibility hooks expected by vLLM's Transformers MM fallback."""
    if hasattr(Qwen3OmniMoeProcessor, "_get_num_multimodal_tokens"):
        return False

    def _get_num_multimodal_tokens(self: Any, image_sizes: Any = None, **_kwargs: Any) -> dict[str, list[int]]:
        # vLLM 0.11 Transformers fallback asks for an image-token estimate during
        # engine profiling. Qwen3OmniMoeProcessor does not expose this helper.
        n_items = len(image_sizes) if image_sizes is not None else 1
        return {"num_image_tokens": [8192] * n_items}

    setattr(Qwen3OmniMoeProcessor, "_get_num_multimodal_tokens", _get_num_multimodal_tokens)  # noqa: B010
    logger.warning("Patched Qwen3OmniMoeProcessor with _get_num_multimodal_tokens for vLLM compatibility.")
    return True


def _version_tuple(version: str) -> tuple[int, int, int]:
    parts = []
    for piece in version.split("."):
        num = ""
        for ch in piece:
            if ch.isdigit():
                num += ch
            else:
                break
        if num:
            parts.append(int(num))
        if len(parts) == 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def _assert_supported_vllm_stack() -> None:
    """Fail fast on known-incompatible vLLM/transformers stacks for Qwen3-Omni."""
    try:
        vllm_ver = metadata.version("vllm")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("vLLM is not installed. Install vllm to use backend: vllm.") from exc
    try:
        tf_ver = metadata.version("transformers")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("transformers is not installed. Install transformers to use backend: vllm.") from exc

    if _version_tuple(vllm_ver) < (0, 22, 0):
        raise RuntimeError(
            "Qwen3-Omni vLLM inference requires vllm>=0.22.0 (working ref: 0.22.1). "
            f"Detected vllm=={vllm_ver}. Upgrade vllm before running eval."
        )
    if _version_tuple(tf_ver) < (4, 57, 3):
        raise RuntimeError(
            "Qwen3-Omni vLLM inference requires transformers>=4.57.3 in practice "
            f"(working ref stack). Detected transformers=={tf_ver}. Upgrade transformers."
        )


def _logprob_value(value: Any) -> float:
    """Coerce a vLLM ``Logprob`` (or a plain float, in older vLLM outputs) to float."""
    return float(value.logprob) if hasattr(value, "logprob") else float(value)


class QwenOmniVLLM:
    """Qwen3-Omni eval wrapper using vLLM tensor parallelism."""

    def __init__(self, config: VllmConfig) -> None:
        self.config = config
        self.llm: LLM | None = None
        self.processor: Any = None
        self._true_token_ids: set[int] = set()
        self._false_token_ids: set[int] = set()

    def load(self) -> QwenOmniVLLM:
        """Load vLLM engine and processor."""
        if LLM is None:
            raise RuntimeError("vLLM is not installed in this environment. Install vllm and retry, or use backend: hf.")
        _assert_supported_vllm_stack()
        cfg = self.config
        tp = cfg.tensor_parallel_size if cfg.tensor_parallel_size is not None else torch.cuda.device_count()
        tp = max(1, tp or 1)

        os.environ["VLLM_USE_V1"] = "1"
        if cfg.max_model_len > 65536:
            os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        # The Qwen3 processor compatibility patch is also registered as a
        # "vllm.general_plugins" entry point, so vLLM re-applies it in every
        # process (engine core + each worker) regardless of start method —
        # no need to force fork to propagate this in-memory monkey patch.
        _patch_qwen3_processor_for_vllm()
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = cfg.worker_multiproc_method

        logger.info("Loading Qwen3-Omni (vLLM): %s  tensor_parallel=%d", cfg.model_id, tp)
        limit_mm = cfg.limit_mm_per_prompt or {"image": 1, "video": 1, "audio": 1}
        mm_kwargs = cfg.mm_processor_kwargs or {"cache_gb": 0}
        self.llm = LLM(
            model=cfg.model_id,
            trust_remote_code=True,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            tensor_parallel_size=tp,
            limit_mm_per_prompt=limit_mm,
            max_num_seqs=cfg.max_num_seqs,
            max_model_len=cfg.max_model_len,
            disable_custom_all_reduce=True,
            enable_prefix_caching=cfg.enable_prefix_caching,
            mm_processor_kwargs=mm_kwargs,
            seed=cfg.seed,
            disable_log_stats=True,
        )
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(cfg.model_id)
        self._true_token_ids, self._false_token_ids = bool_token_ids(self.processor.tokenizer)
        if not self._true_token_ids or not self._false_token_ids:
            logger.warning("Tokenizer has no single-token JSON true/false pair; logprob capture disabled.")
        return self

    def predict(self, clip: ClipData) -> PredictionOutput:
        """Run deepfake detection on one clip via vLLM generate."""
        if self.llm is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        user_text = AV_USER_PROMPT if clip.has_audio else VIDEO_ONLY_USER_PROMPT
        conversation = build_conversation(clip, user_text=user_text, system_prompt=SYSTEM_PROMPT, cfg=self.config)

        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        text = text.replace("<think>\n\n</think>\n\n", "")
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=self.config.use_audio_in_video)

        mm_data: dict[str, Any] = {}
        if audios:
            filtered = [a for a in audios if hasattr(a, "__len__") and len(a) > 0]
            if filtered:
                mm_data["audio"] = filtered
        if images:
            mm_data["image"] = images
        if videos:
            mm_data["video"] = videos

        if SamplingParams is None:
            raise RuntimeError("vLLM SamplingParams unavailable; ensure vllm is installed.")
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=self.config.max_new_tokens,
            logprobs=20,
        )
        outputs = self.llm.generate(
            [  # type: ignore[arg-type]
                {
                    "prompt": text,
                    "multi_modal_data": mm_data,
                    "mm_processor_kwargs": {
                        "max_frames": self.config.video_max_frames,
                        "use_audio_in_video": self.config.use_audio_in_video,
                    },
                }
            ],
            sampling_params=sampling,
        )
        completion = outputs[0].outputs[0]
        raw = completion.text
        prediction = parse_response(raw, has_audio=clip.has_audio, clip_id=clip.clip_id)

        token_ids = list(completion.token_ids)
        raw_step_logprobs = completion.logprobs or []
        step_logprobs = [
            {int(token_id): _logprob_value(value) for token_id, value in position.items()}
            for position in raw_step_logprobs
        ]
        steps = field_value_step_indices(
            token_ids,
            self.processor.tokenizer,
            self._true_token_ids,
            self._false_token_ids,
        )
        video_step = steps.get("video_fake")
        if video_step is not None and video_step < len(step_logprobs):
            video_prob = stream_probs_from_step_logprobs(
                step_logprobs[video_step],
                self._true_token_ids,
                self._false_token_ids,
            )
            if video_prob is not None:
                prediction.video_fake_prob = video_prob
        audio_step = steps.get("audio_fake")
        if clip.has_audio and audio_step is not None and audio_step < len(step_logprobs):
            audio_prob = stream_probs_from_step_logprobs(
                step_logprobs[audio_step],
                self._true_token_ids,
                self._false_token_ids,
            )
            if audio_prob is not None:
                prediction.audio_fake_prob = audio_prob
        return prediction
