"""Two-stage QLoRA SFT: Stage 1 (thinker_lora) then Stage 2 (modality_projectors).

Single pipeline, config/CLI-selected mode — see ``config.py``. No separate
package per stage (that was tried once and reverted; see REBUILD_PLAN.md §3).
"""
