"""Explicit save/load for QLoRA SFT artifacts.

PEFT's ``save_pretrained`` only serializes LoRA adapter weights — Stage 2's
modality projector bridges are regular ``nn.Module`` parameters (not LoRA)
and need explicit save/load logic, kept here rather than relying on PEFT
defaults.

Artifact layout::

    <output_dir>/final_artifact/
        lora/                    PEFT adapter dir (adapter_config.json, adapter weights)
        projectors.safetensors   Stage 2 only: audio_tower.proj1/proj2, visual.merger*
        meta.json                Provenance + reproducibility fields
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from deepfake_moe.train.qlora_sft.config import QLoraSFTConfig
from deepfake_moe.train.qlora_sft.model import _PROJECTOR_MODULE_NAMES, find_named_modules
from deepfake_moe.utils.logging_utils import get_logger


logger = get_logger(__name__)

_LORA_SUBDIR = "lora"
_PROJECTORS_FILE = "projectors.safetensors"
_META_FILE = "meta.json"


def _projector_state_dict(thinker: Any) -> dict[str, torch.Tensor]:
    """Flatten the modality-bridge submodules into one state dict, keyed by name."""
    modules = find_named_modules(thinker, _PROJECTOR_MODULE_NAMES)
    state: dict[str, torch.Tensor] = {}
    for module_name in _PROJECTOR_MODULE_NAMES:
        module = modules.get(module_name)
        if module is None:
            continue
        for param_name, tensor in module.state_dict().items():
            state[f"{module_name}.{param_name}"] = tensor.detach().to("cpu").contiguous()
    return state


def save_artifact(
    model: Any,
    cfg: QLoraSFTConfig,
    output_dir: str | Path,
    *,
    reproducibility: dict[str, Any] | None = None,
) -> Path:
    """Save a stage's trained weights + provenance to ``output_dir``.

    ``model`` is ``build_model(cfg)``'s return value — the thinker submodule
    directly (a ``PeftModel``), not the outer Omni wrapper (see model.py).

    Stage 1 (``thinker_lora``): saves the LoRA adapter only.
    Stage 2 (``modality_projectors``): also saves the trained projector state
    dict. The (frozen, unchanged) inherited LoRA adapter is saved too via the
    same ``model.save_pretrained`` call, so the artifact is self-contained —
    loading it doesn't require also keeping Stage 1's ``init_checkpoint``
    path around.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir / _LORA_SUBDIR))
    logger.info("Saved LoRA adapter -> %s", output_dir / _LORA_SUBDIR)

    if cfg.tuning_mode == "modality_projectors":
        state = _projector_state_dict(model)
        if not state:
            raise RuntimeError("No projector parameters found to save — check _PROJECTOR_MODULE_NAMES in model.py")
        save_file(state, str(output_dir / _PROJECTORS_FILE))
        logger.info("Saved %d projector tensors -> %s", len(state), output_dir / _PROJECTORS_FILE)

    meta = {
        "tuning_mode": cfg.tuning_mode,
        "model_id": cfg.model_id,
        "init_checkpoint": cfg.init_checkpoint,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "lora_target_modules": cfg.lora_target_modules,
        "reproducibility": reproducibility or {},
    }
    (output_dir / _META_FILE).write_text(json.dumps(meta, indent=2))
    logger.info("Saved artifact meta -> %s", output_dir / _META_FILE)
    return output_dir


def load_projector_weights(thinker: Any, artifact_dir: str | Path) -> int:
    """Load a saved projector state dict onto ``thinker``'s bridge submodules, in place.

    Used to resume Stage 2 training, or by ``merge.py`` to apply a completed
    Stage 2 artifact's projector weights before merging. Returns the number
    of tensors loaded.
    """
    artifact_dir = Path(artifact_dir)
    projectors_path = artifact_dir / _PROJECTORS_FILE
    if not projectors_path.exists():
        raise FileNotFoundError(f"No {_PROJECTORS_FILE} in {artifact_dir} — is this a Stage 2 artifact?")

    state = load_file(str(projectors_path))
    modules = find_named_modules(thinker, _PROJECTOR_MODULE_NAMES)
    n_loaded = 0
    for module_name, module in modules.items():
        prefix = f"{module_name}."
        sub_state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        if not sub_state:
            continue
        module.load_state_dict(sub_state, strict=True)
        module.to(dtype=torch.bfloat16)
        n_loaded += len(sub_state)
    if n_loaded == 0:
        raise RuntimeError(f"Loaded 0 projector tensors from {projectors_path} — module names may have changed.")
    logger.info("Loaded %d projector tensors from %s", n_loaded, projectors_path)
    return n_loaded


def load_meta(artifact_dir: str | Path) -> dict[str, Any]:
    """Read an artifact's ``meta.json``."""
    return json.loads((Path(artifact_dir) / _META_FILE).read_text())


def lora_dir(artifact_dir: str | Path) -> Path:
    """Path to the PEFT adapter directory inside an artifact."""
    return Path(artifact_dir) / _LORA_SUBDIR
