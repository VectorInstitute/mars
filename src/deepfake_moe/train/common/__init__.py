"""Shared training machinery.

Modules
-------
targets   Deterministic, GT-only SFT target construction (schema + reason template).
data      Manifest-backed dataset that decodes clips and attaches built targets.
collator  Multimodal collation + label masking (CE on assistant-response tokens only).
splits    Train/Val handling and per-dataset stratified sampling hooks.

Trainer setup and checkpoint selection are stage-specific and live in
``qlora_sft.trainer.SFTTrainer``, not here.
"""
