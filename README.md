# deepfake-moe

[![code checks](https://github.com/VectorInstitute/deepfake-moe/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/deepfake-moe/actions/workflows/code_checks.yml)
[![unit tests](https://github.com/VectorInstitute/deepfake-moe/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/VectorInstitute/deepfake-moe/actions/workflows/unit_tests.yml)
[![integration tests](https://github.com/VectorInstitute/deepfake-moe/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/VectorInstitute/deepfake-moe/actions/workflows/integration_tests.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE.md)
[![Contact](https://img.shields.io/badge/Contact-shaina.raza%40vectorinstitute.ai-green)](mailto:shaina.raza@vectorinstitute.ai)

Audio–video deepfake detection with **Qwen3-Omni**. Phase 1 evaluates the model
zero-shot on public benchmarks (LAV-DF, ArEnAV); the codebase is the traceable
foundation for later Mixture-of-Experts work.

## Overview

- **Model access** — load Qwen3-Omni via HF Transformers (raw forward pass with
encoder/router hooks for inspection) or vLLM (tensor-parallel inference).
- **Evaluation** — prompt the model to classify video/audio as real/fake and
localize fake intervals as JSON, then score against manifest ground truth
(video/audio AUC & F1, temporal IoU).
- **Training** — two-stage QLoRA SFT (Stage 1: thinker LoRA; Stage 2: frozen
LoRA + modality-projector bridges), same launcher for both stages.
- **AVR (Audio-Visual Reliability)** — a self-supervised module that learns
calibrated, per-token cross-modal reliability from the frozen audio/visual
encoders, trained standalone before it feeds evidence tokens into the
Thinker.
- **Configuration** — models live in a single registry and are selected at run
time; results are grouped by experiment and model.



## Structure

```
configs/
  models.yaml            # model registry — one entry per model (hf | vllm)
  eval.yaml              # eval settings; data paths come from .env as ${VAR}
  qlora_train.yaml       # Stage 1 QLoRA SFT training config
  projector_train.yaml   # Stage 2 modality-projector training config
  avr_ssl.yaml            # AVR Stage 1 self-supervised training config
scripts/
  eval.sh                # run eval — local (bash) or Slurm (sbatch; array = shards)
  forward.sh             # run one raw forward pass
  qlora_train.sh         # run QLoRA SFT training — local (bash) or Slurm (sbatch); same for both stages
  avr_ssl_train.sh        # run AVR Stage 1 training — local (bash) or Slurm (sbatch)
  patch_vllm_qwen3.py    # optional vLLM fix for Qwen3-Omni
src/deepfake_moe/
  run_eval.py            # eval entry point
  run_forward.py         # forward entry point
  run_train.py           # QLoRA SFT training entry point
  run_train_avr_ssl.py   # AVR Stage 1 training entry point
  models/                # registry, factory, typed configs, hf + vllm backends; avr/ (AVR module)
  eval/                  # dataset, runner, metrics
  train/                 # common data/collator/targets, two-stage QLoRA SFT (qlora_sft/), AVR Stage 1 (avr_ssl/)
  utils/                 # env/.env loading, logging, media / audio / video I/O
.env.example             # copy to .env; machine paths + FFMPEG_SHARED
```



## Setup

Uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env      # set data paths (and FFMPEG_SHARED for vLLM video)
```

Machine-specific paths live in `.env`; the committed configs reference them as
`${VAR}`, so you never edit paths inside YAML. vLLM video decode requires a
shared FFmpeg 7.x build — point `FFMPEG_SHARED` at it in `.env`.

## Models

Every model is an entry in `configs/models.yaml`. The key is the model name (used
to select the model and to name result folders); `backend` is `hf` or `vllm`.
Add a model by adding an entry — no code changes:

```yaml
my-model:
  backend: vllm
  model_id: "org/checkpoint-name"
  # ... backend-specific settings ...
```



## Run

**Forward pass** (inspection - prints logits, hidden-state, and encoder shapes):

```bash
bash scripts/forward.sh --video clip.mp4 --audio clip.wav --generate
```

**Eval** — the same `scripts/eval.sh` runs locally or on Slurm. Results are
written to `results/<experiment-name>/<model>/`.

```bash
# local smoke test
bash scripts/eval.sh --dry-run

# single Slurm job
sbatch scripts/eval.sh --experiment-name Qwen3-Baseline --model qwen3-omni

# sharded across a Slurm array (shards derived from the array size)
sbatch --array=0-4 scripts/eval.sh --experiment-name Qwen3-Baseline --model qwen3-omni
```

Metrics are aggregated automatically: a single job writes them at the end, and a
sharded array queues a dependent (afterok) job that combines all shards into one
report. To (re)compute metrics from existing shard predictions manually:

```bash
bash scripts/eval.sh --metrics-only --experiment-name Qwen3-Baseline --model qwen3-omni
```

Re-run only the clips that failed (load/predict/parse errors) in a prior run:

```bash
bash scripts/eval.sh --retry-failed --experiment-name Qwen3-Baseline --model qwen3-omni
```

If you hit the vLLM Qwen3-Omni `cu_seqlens` bug, apply the patch once:

```bash
uv run python scripts/patch_vllm_qwen3.py
```

**Training** — two-stage QLoRA SFT, same `scripts/qlora_train.sh` launcher for
both stages, local or Slurm. Stage 2 requires Stage 1's `final_artifact/` as
`init_checkpoint` in `configs/projector_train.yaml`.

```bash
# Stage 1 — thinker LoRA
sbatch scripts/qlora_train.sh --config configs/qlora_train.yaml

# Stage 2 — modality projectors (frozen Stage-1 LoRA)
sbatch scripts/qlora_train.sh --config configs/projector_train.yaml
```

Run `--smoke-test` (tiny, validates the pipeline end-to-end) before every real
launch, and `--profile` (10 real-sized steps) to size the Slurm `--time`:

```bash
bash scripts/qlora_train.sh --config configs/qlora_train.yaml --smoke-test
bash scripts/qlora_train.sh --config configs/qlora_train.yaml --profile
```

Merge a finished stage's LoRA (+ projectors, if Stage 2) into a standalone
BF16 model for eval:

```bash
uv run deepfake-moe-qlora-merge \
    --config configs/qlora_train.yaml \
    --artifact-dir checkpoints/<run_name>/final_artifact
```

**AVR Stage 1** — self-supervised pretraining of the AVR module's imputation,
router, and expert weights (no labels, no Thinker). Standalone: it doesn't
depend on QLoRA SFT and doesn't touch `train/qlora_sft/`. Run via
`scripts/avr_ssl_train.sh`, local or Slurm:

```bash
sbatch scripts/avr_ssl_train.sh --config configs/avr_ssl.yaml
```

Same `--smoke-test` / `--profile` convention as QLoRA training:

```bash
bash scripts/avr_ssl_train.sh --config configs/avr_ssl.yaml --smoke-test
bash scripts/avr_ssl_train.sh --config configs/avr_ssl.yaml --profile
```

Checkpoints (best + final) land at `checkpoints/<run_name>/{best,final}_artifact/`
as a plain `avr_model.pt` state dict (imputation, router, and expert weights
only) plus `meta.json`.



## Development

```bash
uv run pytest -m "not integration_test"
```

See `docs/user_guide.md` for tooling (pre-commit, mypy, docs, CI).
