"""LoRA -> BF16 merge for a finalized QLoRA SFT artifact.

Loads the base model in full BF16 (not NF4 — merging needs full-precision
weights to fold the LoRA delta in), applies the artifact's LoRA adapter via
``PeftModel.merge_and_unload()``, applies any saved projector weights on top
(Stage 2 artifacts), and saves the resulting standalone model + processor.

``configs/models.yaml`` already registers ``qwen3-omni-qlora-sft`` pointing at
``${QLORA_SFT_MERGED_DIR}`` — this writes there by default so that entry
resolves without any YAML edit.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import Qwen3OmniMoeForConditionalGeneration

from deepfake_moe.train.qlora_sft.artifacts import load_meta, load_projector_weights, lora_dir
from deepfake_moe.train.qlora_sft.config import QLoraSFTConfig
from deepfake_moe.train.qlora_sft.model import (
    load_processor,
    patch_peft_weight_converter_kwargs,
    patch_thinker_text_no_split_modules,
)
from deepfake_moe.utils.env import load_env
from deepfake_moe.utils.logging_utils import get_logger


logger = get_logger(__name__)


def load_bf16_base(model_id: str, attn_implementation: str | None) -> Any:
    """Load the full Omni model in plain BF16 (no quantization) as a merge target."""
    patch_thinker_text_no_split_modules()
    kwargs: dict[str, Any] = {"device_map": "auto", "dtype": torch.bfloat16}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    logger.info("Loading BF16 base model for merge: %s", model_id)
    return Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_id, **kwargs)


def merge_artifact(cfg: QLoraSFTConfig, artifact_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    """Merge one artifact's LoRA (+ projectors, if any) into a standalone BF16 model."""
    load_env()
    artifact_dir = Path(artifact_dir)
    meta = load_meta(artifact_dir)
    logger.info("Merging artifact: tuning_mode=%s  model_id=%s", meta["tuning_mode"], meta["model_id"])

    base = load_bf16_base(meta["model_id"], cfg.attn_implementation)
    # LoRA was trained against the FULL model (paths prefixed `thinker.`), so
    # the adapter must be loaded onto the full model here too, not the
    # extracted `.thinker` submodule.
    patch_peft_weight_converter_kwargs()
    base = PeftModel.from_pretrained(base, str(lora_dir(artifact_dir)))
    base = base.merge_and_unload()
    logger.info("Merged LoRA into base weights.")

    projectors_path = artifact_dir / "projectors.safetensors"
    if projectors_path.exists():
        n_loaded = load_projector_weights(base, artifact_dir)
        logger.info("Applied %d projector tensors from Stage 2 artifact.", n_loaded)

    resolved_output = Path(output_dir) if output_dir else Path(os.environ.get("QLORA_SFT_MERGED_DIR", ""))
    if not str(resolved_output) or str(resolved_output) == ".":
        raise ValueError(
            "No --output-dir given and QLORA_SFT_MERGED_DIR is not set in .env — "
            "configs/models.yaml's qwen3-omni-qlora-sft entry resolves model_id from that variable."
        )
    resolved_output.mkdir(parents=True, exist_ok=True)
    base.save_pretrained(str(resolved_output))
    processor = load_processor(cfg)
    processor.save_pretrained(str(resolved_output))
    logger.info("Saved merged model + processor -> %s", resolved_output)
    logger.info(
        "configs/models.yaml already registers this as 'qwen3-omni-qlora-sft' "
        '(model_id: "${QLORA_SFT_MERGED_DIR}") — no YAML edit needed if that env var matches.'
    )
    return resolved_output


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Merge a QLoRA SFT artifact's LoRA (+ projectors) into a standalone BF16 model"
    )
    p.add_argument("--config", required=True, help="Same qlora_train.yaml/projector_train.yaml used to train it")
    p.add_argument("--artifact-dir", required=True, help="Path to the stage's final_artifact/ directory")
    p.add_argument("--output-dir", default=None, help="Defaults to $QLORA_SFT_MERGED_DIR from .env")
    return p.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    cfg = QLoraSFTConfig.from_yaml(args.config)
    merge_artifact(cfg, args.artifact_dir, args.output_dir)


if __name__ == "__main__":
    main()
