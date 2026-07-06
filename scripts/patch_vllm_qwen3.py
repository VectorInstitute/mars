#!/usr/bin/env python
"""Patch vLLM 0.22.x for Qwen3-Omni cu_seqlens CUDA tensor bug.

Adapted from the working sonic-o1-mllm environment setup.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_target_file() -> Path:
    # Imported lazily so the script fails gracefully when vllm is not installed.
    import vllm  # noqa: PLC0415

    vllm_root = Path(vllm.__file__).parent
    target = vllm_root / "model_executor" / "layers" / "attention" / "mm_encoder_attention.py"
    if not target.exists():
        raise FileNotFoundError(f"Target file not found: {target}")
    return target


def _apply_patch(path: Path) -> str:
    content = path.read_text()
    if "PATCH: ensure cu_seqlens is a CUDA tensor" in content:
        return "already_applied"

    old = (
        "        query, key, value = self.view_qkv_to_4d(query, key, value, bsz, q_len, kv_len)\n\n"
        "        output = vit_flash_attn_wrapper("
    )
    new = (
        "        query, key, value = self.view_qkv_to_4d(query, key, value, bsz, q_len, kv_len)\n\n"
        "        # PATCH: ensure cu_seqlens is a CUDA tensor (Qwen3-Omni + vLLM 0.22.x)\n"
        "        if cu_seqlens is not None and not isinstance(cu_seqlens, torch.Tensor):\n"
        "            import numpy as np\n"
        "            cu_seqlens = torch.from_numpy(np.asarray(cu_seqlens, dtype=np.int32)).to(query.device)\n"
        "        elif cu_seqlens is not None:\n"
        "            cu_seqlens = cu_seqlens.to(query.device)\n"
        "        output = vit_flash_attn_wrapper("
    )
    if old not in content:
        raise RuntimeError("Patch anchor not found. vLLM source may differ from expected 0.22.x layout.")
    path.write_text(content.replace(old, new))
    return "patched"


def main() -> None:
    """Locate the installed vLLM and apply the Qwen3-Omni cu_seqlens patch."""
    try:
        target = _find_target_file()
        state = _apply_patch(target)
        if state == "already_applied":
            print(f"Patch already applied: {target}")
        else:
            print(f"Patch applied successfully: {target}")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
