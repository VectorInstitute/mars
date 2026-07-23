"""Throwaway diagnostic for train/qlora_sft/model.py — NOT part of the pipeline.

Loads Stage 1 (thinker_lora), checks that LoRA landed only on the text
decoder's self_attn.{q_proj,v_proj} (not audio_tower's identically-named
attention), saves the adapter to a temp dir, then loads Stage 2
(modality_projectors) from it and checks the projector bridges are BF16 and
trainable while everything else is frozen.

Usage:
    uv run python scripts/smoke_test_qlora_model.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepfake_moe.train.qlora_sft.config import QLoraSFTConfig  # noqa: E402
from deepfake_moe.train.qlora_sft.model import build_stage1_model, build_stage2_model, find_named_modules  # noqa: E402


def _report_lora_placement(model) -> None:
    print("\n--- LoRA placement check ---")
    lora_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    n_decoder = sum(1 for n in lora_param_names if ".audio_tower." not in n and ".visual." not in n)
    n_audio = sum(1 for n in lora_param_names if ".audio_tower." in n)
    n_visual = sum(1 for n in lora_param_names if ".visual." in n)
    print(f"trainable params total: {len(lora_param_names)}")
    print(f"  in text decoder      : {n_decoder}  (expected: > 0)")
    print(f"  in audio_tower       : {n_audio}  (expected: 0 — LoRA must NOT touch the audio encoder)")
    print(f"  in visual            : {n_visual}  (expected: 0 — LoRA must NOT touch the vision encoder)")
    print("sample trainable param names:")
    for n in lora_param_names[:6]:
        print(f"  {n}")
    if n_audio or n_visual:
        print("!!! FAIL: LoRA leaked into an encoder tower — target_modules regex is wrong.")
    else:
        print("OK: LoRA is decoder-only.")


def _report_dtypes(model, names: tuple[str, ...], label: str) -> None:
    print(f"\n--- {label} dtype/trainable check ---")
    named = find_named_modules(model, names)
    for name in names:
        module = named.get(name)
        if module is None:
            print(f"  {name}: NOT FOUND (module path may have changed upstream)")
            continue
        params = list(module.parameters())
        dtypes = {str(p.dtype) for p in params}
        trainable = all(p.requires_grad for p in params) if params else False
        print(f"  {name}: dtype={dtypes} trainable={trainable} n_params={sum(p.numel() for p in params)}")


def main() -> None:
    """Run the Stage 1 -> save -> Stage 2 smoke test and print diagnostics."""
    stage1_cfg = QLoraSFTConfig(tuning_mode="thinker_lora")
    print(f"Loading Stage 1 base model: {stage1_cfg.model_id}")
    model1 = build_stage1_model(stage1_cfg)
    _report_lora_placement(model1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        adapter_dir = Path(tmp_dir) / "stage1_adapter"
        model1.save_pretrained(str(adapter_dir))
        print(f"\nSaved Stage 1 adapter to {adapter_dir} for Stage 2 load test.")
        del model1
        torch.cuda.empty_cache()

        stage2_cfg = QLoraSFTConfig(tuning_mode="modality_projectors", init_checkpoint=str(adapter_dir))
        print(f"\nLoading Stage 2 base model with init_checkpoint={adapter_dir}")
        model2 = build_stage2_model(stage2_cfg)
        _report_dtypes(
            model2,
            ("audio_tower.proj1", "audio_tower.proj2", "visual.merger", "visual.merger_list"),
            "Stage 2 projector bridges",
        )
        n_trainable = sum(p.numel() for p in model2.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model2.parameters())
        print(f"\nStage 2 trainable / total params: {n_trainable} / {n_total}")

    print("\nDone. Review the OK/FAIL lines above.")


if __name__ == "__main__":
    main()
