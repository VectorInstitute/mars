"""Model construction for the two-stage QLoRA SFT pipeline.

Both builders load the Thinker class DIRECTLY
(``Qwen3OmniMoeThinkerForConditionalGeneration``) rather than the outer
composite ``Qwen3OmniMoeForConditionalGeneration``. The Thinker is the actual
training target — it owns the LM head and a real ``forward()`` that returns a
loss — whereas the composite only orchestrates ``.generate()`` and has no
trainable forward of its own. Loading the Thinker directly has three payoffs:

1. The talker + code2wav stacks never enter GPU memory (they load as
   ``UNEXPECTED`` keys against this class and are dropped), so the footprint
   is smaller than a composite load.
2. ``from_pretrained(..., device_map="auto")`` returns the exact object we
   train, so ``hf_device_map`` sits on the training target itself — no need
   to copy the bookkeeping dict onto an extracted submodule. HF ``Trainer``
   then recognizes the model as already multi-GPU model-parallel and skips
   its own single-device placement.
3. PEFT wraps the Thinker directly, so ``Trainer(model=...)`` receives a
   ``PeftModel`` as required.

Target module paths are resolved from the Thinker's own ``named_modules()``
(no ``thinker.`` prefix — e.g. ``model.layers.0.self_attn.q_proj``); PEFT
suffix-matches on save/merge, so the adapter still loads onto the composite in
``merge.py``.

Stage 1 (``thinker_lora``): NF4-quantized text decoder + LoRA adapters on
``model.layers.*.self_attn.{q_proj,v_proj}`` only. The audio/visual towers
are skipped from quantization (stay BF16) but remain frozen in Stage 1.

Stage 2 (``modality_projectors``): loads Stage 1's LoRA (frozen), then trains
only the modality bridges — ``audio_tower.proj1/proj2``,
``visual.merger``/``merger_list`` — kept explicitly in BF16.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    Qwen3OmniMoeProcessor,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

from deepfake_moe.train.qlora_sft.config import QLoraSFTConfig
from deepfake_moe.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Anchored to the TEXT DECODER only (`*.model.layers.N.self_attn.*` — the
# leading `(?:.*\.)?` tolerates the `thinker.` prefix from resolving against
# the full model rather than the extracted thinker submodule).
# audio_tower.layers.N.self_attn.{q_proj,v_proj} shares the same leaf names
# and would otherwise be matched too, silently expanding Stage 1 ("Thinker
# QLoRA") into the audio encoder — which Stage 2 must later own exclusively.
_LORA_TARGET_MODULE_TEMPLATE = r"(?:.*\.)?model\.layers\.\d+\.self_attn\.({modules})"

# Skipped from NF4 quantization: the audio/visual towers (so their bridge
# projectors can be finetuned in Stage 2 — 4-bit weights aren't directly
# trainable) and lm_head (numerically stable logits). Matched as a substring
# of the full dotted module path by transformers' bnb quantizer, so this
# works regardless of nesting depth (actual path is `thinker.audio_tower...`).
_QUANT_SKIP_MODULES = ["audio_tower", "visual", "lm_head"]

# Stage 2's trainable set — the modality projector bridges only.
_PROJECTOR_MODULE_NAMES = ("audio_tower.proj1", "audio_tower.proj2", "visual.merger", "visual.merger_list")


def _resolve_lora_target_names(thinker: Any, target_modules: list[str]) -> list[str]:
    """Resolve the exact dotted module names to hand ``LoraConfig`` as an explicit list.

    Deliberately does NOT pass a regex string to ``target_modules`` — on the
    installed PEFT version this was observed to run ``set(target_modules)``
    on the string, silently splitting it into individual characters instead
    of using it as a ``re.fullmatch`` pattern (raised "Target modules
    {'.', '|', 'n', ...} not found"). Resolving names ourselves and passing
    an explicit ``list[str]`` sidesteps that string-handling behavior
    entirely, and lets us log/assert on the exact match count.
    """
    modules = "|".join(re.escape(m) for m in target_modules)
    pattern = re.compile(_LORA_TARGET_MODULE_TEMPLATE.format(modules=modules))
    names = [name for name, _ in thinker.named_modules() if pattern.fullmatch(name)]
    if not names:
        raise RuntimeError(
            f"No modules matched {pattern.pattern!r} under {type(thinker).__name__} "
            "— check target module naming against the installed transformers version."
        )
    return names


def load_processor(cfg: QLoraSFTConfig) -> Qwen3OmniMoeProcessor:
    """Load the shared processor (tokenizer + audio/vision feature extractors)."""
    return Qwen3OmniMoeProcessor.from_pretrained(cfg.model_id)


def patch_thinker_text_no_split_modules() -> None:
    """Work around a ``transformers`` bug that can split a decoder layer across GPUs.

    ``Qwen3OmniMoeThinkerTextModel`` (the actual 30B-A3B MoE text decoder)
    inherits ``_no_split_modules`` from the generic ``Qwen3OmniMoePreTrainedModel``
    base — ``["Qwen3OmniMoeDecoderLayer", "Qwen3OmniMoeVisionBlock"]``, where
    ``Qwen3OmniMoeDecoderLayer`` isn't even a real class in this model — instead
    of from the sibling ``Qwen3OmniMoeThinkerTextPreTrainedModel``, which
    correctly lists ``["Qwen3OmniMoeThinkerTextDecoderLayer"]`` but is never
    actually used as a base class anywhere. Composite-model ``_no_split_modules``
    aggregation (``PreTrainedModel.post_init``) walks up from this wrong value,
    so ``accelerate.infer_auto_device_map`` has no signal that a decoder
    layer's ``self_attn`` and its (huge) MoE ``mlp`` block must stay on one
    GPU — it can and does place them on different devices, which crashes the
    layer's own residual add (``residual + hidden_states``, cross-device)
    during training forward. Patching the class attribute before
    ``from_pretrained`` runs fixes this at the source: instances built
    afterward pick up the correct value during their own ``post_init``.
    """
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextModel  # noqa: PLC0415

    Qwen3OmniMoeThinkerTextModel._no_split_modules = ["Qwen3OmniMoeThinkerTextDecoderLayer"]


def patch_peft_weight_converter_kwargs() -> None:
    """Work around a peft/transformers ``WeightConverter`` signature mismatch.

    peft 0.19.1's MoE adapter-loading path (triggered by any architecture with
    a registered checkpoint-conversion mapping, incl. our LoRA-only q_proj/
    v_proj adapter on this MoE model) calls ``WeightConverter(...,
    distributed_operation=..., quantization_operation=...)``, but installed
    transformers only accepts those as post-init attributes, raising
    ``TypeError``. Already fixed on peft's git main (construct, then assign);
    not yet released. Remove once peft picks up the fix.
    """
    from transformers.core_model_loading import WeightConverter  # noqa: PLC0415

    if getattr(WeightConverter.__init__, "_deepfake_moe_patched", False):
        return
    original_init = WeightConverter.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        distributed_operation = kwargs.pop("distributed_operation", None)
        quantization_operation = kwargs.pop("quantization_operation", None)
        original_init(self, *args, **kwargs)
        if distributed_operation is not None:
            self.distributed_operation = distributed_operation
        if quantization_operation is not None:
            self.quantization_operation = quantization_operation

    patched_init._deepfake_moe_patched = True  # type: ignore[attr-defined]
    WeightConverter.__init__ = patched_init  # type: ignore[method-assign]


def _max_memory_map(headroom_fraction: float) -> dict[int, int] | None:
    """Cap each visible GPU's ``device_map="auto"`` weight budget, in bytes."""
    if not torch.cuda.is_available():
        return None
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        return None
    budget_fraction = 1.0 - headroom_fraction
    return {i: int(torch.cuda.get_device_properties(i).total_memory * budget_fraction) for i in range(n_gpus)}


def _load_quantized_base(cfg: QLoraSFTConfig) -> Any:
    """Load the Thinker directly: text decoder in NF4, towers BF16, sharded across GPUs.

    Loads ``Qwen3OmniMoeThinkerForConditionalGeneration`` (NOT the outer
    composite). Talker + code2wav weights load as ``UNEXPECTED`` against this
    class and are dropped, so they never occupy GPU memory. The returned
    object is the training target and carries ``hf_device_map`` directly,
    which is what lets HF ``Trainer`` treat it as already model-parallel.
    """
    patch_thinker_text_no_split_modules()
    quant_config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=_QUANT_SKIP_MODULES,
    )
    kwargs: dict[str, Any] = {
        "quantization_config": quant_config,
        "device_map": "auto",
        "dtype": torch.bfloat16,
    }
    max_memory = _max_memory_map(cfg.gpu_memory_headroom_fraction)
    if max_memory:
        kwargs["max_memory"] = max_memory
        logger.info(
            "Per-GPU max_memory budget (bytes, %d%% headroom reserved): %s",
            int(cfg.gpu_memory_headroom_fraction * 100),
            max_memory,
        )
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation
    logger.info("Loading Thinker directly (NF4 decoder, BF16 towers), sharded across GPUs: %s", cfg.model_id)
    thinker = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(cfg.model_id, **kwargs)
    dev_map = getattr(thinker, "hf_device_map", None)
    if dev_map:
        logger.info("Thinker device map spans: %s", sorted({str(d) for d in dev_map.values()}))
    return thinker


def _freeze_all(model: Any) -> None:
    for param in model.parameters():
        param.requires_grad = False


def find_named_modules(model: Any, names: tuple[str, ...]) -> dict[str, Any]:
    """Resolve bare module names (e.g. ``"audio_tower.proj1"``) to their submodules.

    Matches on a dotted-path suffix rather than an exact/prefix name, so this
    is robust to any wrapper/nesting prefix — e.g. once ``get_peft_model``
    wraps the full Omni model, the real path is
    ``base_model.model.thinker.audio_tower.proj1``, not the bare name. Used
    for both the trainable/dtype setup here and for artifact save/load
    (``artifacts.py``), so both agree on module identity.
    """
    found: dict[str, Any] = {}
    remaining = set(names)
    for module_name, module in model.named_modules():
        matched = {target for target in remaining if module_name == target or module_name.endswith(f".{target}")}
        for target in matched:
            found[target] = module
        remaining -= matched
        if not remaining:
            break
    return found


def _set_trainable_by_name(model: Any, names: tuple[str, ...]) -> int:
    """Unfreeze all parameters of the named submodules (see ``find_named_modules``)."""
    modules = find_named_modules(model, names)
    missing = set(names) - set(modules)
    if missing:
        raise RuntimeError(f"Could not resolve module(s) {sorted(missing)} on {type(model).__name__}")
    n_trainable = 0
    for module in modules.values():
        for param in module.parameters():
            param.requires_grad = True
            n_trainable += param.numel()
    return n_trainable


def _cast_modules_to_bf16(model: Any, names: tuple[str, ...]) -> None:
    """Pin projector submodules to BF16.

    They're already real (non-quantized) parameters — ``_QUANT_SKIP_MODULES``
    keeps them out of NF4 — this just guards against a stray dtype mismatch
    silently downcasting a trainable projector: 4-bit base weights can't host
    a trainable projector, so this can't rely on PEFT's defaults.
    """
    for module in find_named_modules(model, names).values():
        module.to(dtype=torch.bfloat16)


def build_stage1_model(cfg: QLoraSFTConfig) -> Any:
    """Stage 1: NF4 decoder + LoRA on the text decoder's attention projections only.

    Loads, prepares, LoRA-wraps, and returns the Thinker directly.
    """
    thinker = _load_quantized_base(cfg)
    # Gradient checkpointing OFF: it's incompatible with the lm_head pre-hook in
    # train.py (which slices hidden states to the answer positions before the
    # head), and its use_cache=False side-effect is what broke eval generation.
    # Memory is instead kept in check by NF4 4-bit + the pre-hook avoiding the
    # full-sequence 150k-vocab logit tensor. See train.py's _masked_lm_loss.
    thinker = prepare_model_for_kbit_training(  # type: ignore[no-untyped-call]
        thinker, use_gradient_checkpointing=False
    )

    target_names = _resolve_lora_target_names(thinker, cfg.lora_target_modules)
    logger.info("LoRA target modules resolved: %d  (e.g. %s)", len(target_names), target_names[:2])
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_names,
        bias="none",
        task_type="CAUSAL_LM",
    )
    thinker = get_peft_model(thinker, lora_config)
    thinker.print_trainable_parameters()
    return thinker


def build_stage2_model(cfg: QLoraSFTConfig) -> Any:
    """Stage 2: load + freeze Stage 1's LoRA, train only the modality projector bridges.

    Requires ``cfg.init_checkpoint`` (validated in ``QLoraSFTConfig.__post_init__``).
    Loads and returns the Thinker directly.
    """
    if cfg.init_checkpoint is None:
        raise ValueError("Stage 2 requires cfg.init_checkpoint (see QLoraSFTConfig.__post_init__)")
    from deepfake_moe.train.qlora_sft.artifacts import lora_dir  # noqa: PLC0415 (avoid module-level cycle)

    patch_peft_weight_converter_kwargs()
    thinker = _load_quantized_base(cfg)
    thinker = PeftModel.from_pretrained(thinker, str(lora_dir(cfg.init_checkpoint)), is_trainable=False)

    _freeze_all(thinker)
    n_trainable = _set_trainable_by_name(thinker, _PROJECTOR_MODULE_NAMES)
    if n_trainable == 0:
        raise RuntimeError(
            f"No trainable parameters found under {_PROJECTOR_MODULE_NAMES} — "
            "module names may have changed upstream; update _PROJECTOR_MODULE_NAMES in model.py."
        )
    _cast_modules_to_bf16(thinker, _PROJECTOR_MODULE_NAMES)
    logger.info("Stage 2: %d trainable projector parameters (BF16)", n_trainable)
    return thinker


def build_model(cfg: QLoraSFTConfig) -> Any:
    """Dispatch on ``cfg.tuning_mode`` to the Stage 1 / Stage 2 model builder."""
    if cfg.tuning_mode == "thinker_lora":
        return build_stage1_model(cfg)
    if cfg.tuning_mode == "modality_projectors":
        return build_stage2_model(cfg)
    raise ValueError(f"Unknown tuning_mode: {cfg.tuning_mode!r}")
