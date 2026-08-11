"""Device routing for Qwen3-Omni models split across multiple GPUs via ``device_map``.

Extracted from ``QwenOmni`` (eval) so ``train/qlora_sft/trainer.py``'s
``SFTTrainer`` can reuse the exact same "which GPU does this input tensor go
to" logic instead of a second, unverified implementation. ``QwenOmni``'s own
methods now delegate here — see that module.

For an Accelerate-sharded model (``device_map`` spans >1 device), inputs must
all enter the model on ONE anchor device (the embedding layer's device);
splitting them per-modality across devices can trigger internal concat/ops
on mixed devices. A single-device model instead routes each tensor to its
own modality's device (harmless either way, kept for parity with a
non-sharded load).
"""

from __future__ import annotations

from typing import Any

import torch

from deepfake_moe.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Dedup the "sharded device_map detected" log line per model instance rather
# than spamming it on every batch.
_logged_sharded_routing: set[int] = set()


def parse_hf_device(dev: Any) -> torch.device:
    """Normalize one entry of an HF ``hf_device_map`` into a ``torch.device``."""
    if isinstance(dev, torch.device):
        return dev
    if isinstance(dev, int):
        return torch.device(f"cuda:{dev}" if dev >= 0 else "cpu")
    if isinstance(dev, str):
        return torch.device(dev)
    if isinstance(dev, (list, tuple)) and dev:
        return parse_hf_device(dev[0])
    raise TypeError(f"Unsupported device entry in hf_device_map: {dev!r}")


def module_device(module: Any) -> torch.device:
    """Return the device of a module's first parameter, defaulting to CPU."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _device_from_hf_map(hf_map: dict[str, Any], *name_fragments: str) -> torch.device | None:
    for name, dev in hf_map.items():
        if any(fragment in name for fragment in name_fragments):
            return parse_hf_device(dev)
    return None


def thinker_input_devices(model: Any, module: Any) -> dict[str, torch.device]:
    """Resolve per-modality anchor devices (default/visual/audio) for ``model``."""
    root = model if model is not None else module
    thinker = getattr(root, "thinker", module)
    hf_map = getattr(root, "hf_device_map", None) or getattr(module, "hf_device_map", None) or {}

    embed = getattr(getattr(thinker, "model", None), "embed_tokens", None)
    default = (
        _device_from_hf_map(hf_map, "thinker.model.embed_tokens", "model.embed_tokens", "embed_tokens")
        or (module_device(embed) if embed is not None else None)
        or module_device(thinker)
    )
    visual_mod = getattr(thinker, "visual", None)
    audio_mod = getattr(thinker, "audio_tower", None)
    visual = _device_from_hf_map(hf_map, "thinker.visual", ".visual") or (
        module_device(visual_mod) if visual_mod is not None else default
    )
    audio = _device_from_hf_map(hf_map, "thinker.audio_tower", "audio_tower") or (
        module_device(audio_mod) if audio_mod is not None else default
    )
    return {"default": default, "visual": visual, "audio": audio}


def is_sharded_hf_model(model: Any, module: Any) -> bool:
    """Return True when Accelerate dispatched submodules across multiple devices."""
    root = model if model is not None else module
    hf_map = getattr(root, "hf_device_map", None) or getattr(module, "hf_device_map", None) or {}
    if not hf_map:
        return False
    devices = {str(parse_hf_device(dev)) for dev in hf_map.values()}
    return len(devices) > 1


def tensor_device(key: str, devices: dict[str, torch.device]) -> torch.device:
    """Pick the anchor device for one input tensor by its processor key name."""
    key_lower = key.lower()
    if key_lower in {"input_features", "feature_attention_mask", "audio_feature_lengths"} or "audio" in key_lower:
        return devices["audio"]
    if any(token in key_lower for token in ("video", "pixel", "image", "grid_thw")):
        return devices["visual"]
    return devices["default"]


def prepare_inputs(model: Any, inputs: Any, *, module: Any | None = None) -> Any:
    """Move processor inputs (in place) to the right device(s) for ``model``.

    ``model`` should be the object carrying ``hf_device_map`` (whatever
    ``from_pretrained(device_map=...)`` returned, even if wrapped by PEFT
    afterwards — PEFT forwards unknown attributes through). ``module``
    defaults to ``model`` itself; pass a different submodule (e.g. the
    thinker) when that's what actually owns the layers being routed for.
    """
    module = module if module is not None else model
    devices = thinker_input_devices(model, module)
    dtype = getattr(module, "dtype", torch.float16)

    if is_sharded_hf_model(model, module):
        target = devices["default"]
        if id(model) not in _logged_sharded_routing:
            logger.info("HF sharded device_map detected; routing all model inputs to %s", target)
            _logged_sharded_routing.add(id(model))
        for key in list(inputs.keys()):
            value = inputs[key]
            if not isinstance(value, torch.Tensor):
                continue
            inputs[key] = value.to(device=target, dtype=dtype) if value.is_floating_point() else value.to(device=target)
        return inputs

    for key in list(inputs.keys()):
        value = inputs[key]
        if not isinstance(value, torch.Tensor):
            continue
        dev = tensor_device(key, devices)
        inputs[key] = value.to(device=dev, dtype=dtype) if value.is_floating_point() else value.to(device=dev)
    return inputs
