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
- **Configuration** — models live in a single registry and are selected at run
time; results are grouped by experiment and model.



## Structure

```
configs/
  models.yaml            # model registry — one entry per model (hf | vllm)
  eval.yaml              # eval settings; data paths come from .env as ${VAR}
scripts/
  eval.sh               # run eval — local (bash) or Slurm (sbatch; array = shards)
  forward.sh            # run one raw forward pass
  patch_vllm_qwen3.py   # optional vLLM fix for Qwen3-Omni
src/deepfake_moe/
  run_eval.py           # eval entry point
  run_forward.py        # forward entry point
  models/               # registry, factory, typed configs, hf + vllm backends
  eval/                 # dataset, runner, metrics
  utils/                # env/.env loading, logging, media / audio / video I/O
.env.example            # copy to .env; machine paths + FFMPEG_SHARED
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

**Forward pass** (inspection — prints logits, hidden-state, and encoder shapes):

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



## Development

```bash
uv run pytest -m "not integration_test"
```

See `docs/user_guide.md` for tooling (pre-commit, mypy, docs, CI).
