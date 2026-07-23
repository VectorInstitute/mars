#!/bin/bash
# Two-stage QLoRA SFT training. The same file runs locally or on Slurm:
#
#   bash   scripts/qlora_train.sh --config configs/qlora_train.yaml --smoke-test
#   bash   scripts/qlora_train.sh --config configs/qlora_train.yaml --profile
#   sbatch scripts/qlora_train.sh --config configs/qlora_train.yaml
#   sbatch scripts/qlora_train.sh --config configs/projector_train.yaml
#
# --smoke-test and --profile apply canned --set overrides (below) and are
# stripped before forwarding the rest to `python -m deepfake_moe.run_train`.
# Any --set key=value you
# pass directly is forwarded as-is; run the smoke test before every real
# launch, and the profiling run before committing to a Slurm --time.
#
#SBATCH --job-name=deepfake_moe_qlora_train
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/qlora_train_%j.out
#SBATCH --error=logs/qlora_train_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p logs

# Machine config; on a cluster also load runtime modules + shared FFmpeg.
# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; source .env; set +a; }
if command -v module >/dev/null 2>&1; then
  module load StdEnv/2023 gcc/12.3 cuda/12.2
  export CUDA_HOME="${CUDA_ROOT:-${CUDA_HOME:-}}"
fi
if [[ -n "${FFMPEG_SHARED:-}" && -d "${FFMPEG_SHARED}/lib" ]]; then
  export PATH="${FFMPEG_SHARED}/bin:${PATH}"
  export LD_LIBRARY_PATH="${FFMPEG_SHARED}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
# flash-attn is an opt-in uv dependency group (only resolvable via the
# Compute Canada / DRAC wheelhouse, see pyproject.toml) — sync it here so
# Stage 1's flash_attention_2 is available, when that wheelhouse exists.
if [[ -d /cvmfs/soft.computecanada.ca ]]; then
  uv sync --group flash-attn
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

args=()
extra_set=()
for arg in "$@"; do
  case "$arg" in
    --smoke-test)
      # Smoke-test recipe: confirm the job completes end-to-end, checkpoint
      # has the right tensor count for the mode, best (lowest) val_loss logged,
      # W&B synced — run this before every real launch. Generation/parse-rate
      # is not checked here; verify it separately via the vLLM eval harness on
      # the resulting checkpoint.
      extra_set+=(
        --set train_samples_per_dataset=1 --set val_samples_per_dataset=1
        --set max_steps=1 --set gradient_accumulation_steps=1
        --set eval_steps=1 --set save_steps=1
        --set num_frames=4
      )
      ;;
    --profile)
      # Profiling recipe: 10 steps, real num_frames, no eval/save/grad-accum —
      # use train_runtime / 10 to size the Slurm --time. Not representative of
      # the 4-frame smoke test (real run adds audio + full frame count).
      extra_set+=(
        --set max_steps=10 --set num_frames=64
        --set eval_strategy=no --set save_strategy=no
        --set load_best_model_at_end=no --set gradient_accumulation_steps=1
      )
      ;;
    *)
      args+=("$arg")
      ;;
  esac
done

uv run python -m deepfake_moe.run_train "${args[@]}" "${extra_set[@]}"
