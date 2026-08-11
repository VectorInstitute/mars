#!/bin/bash
# AVR Stage 1: self-supervised pretraining of imputation/router/experts.
# The same file runs locally or on Slurm:
#
#   bash   scripts/avr_ssl_train.sh --config configs/avr_ssl.yaml --smoke-test
#   bash   scripts/avr_ssl_train.sh --config configs/avr_ssl.yaml --profile
#   sbatch scripts/avr_ssl_train.sh --config configs/avr_ssl.yaml
#
# --smoke-test and --profile apply canned --set overrides (below) and are
# stripped before forwarding the rest to
# `python -m deepfake_moe.run_train_avr_ssl`. Any --set key=value you pass
# directly is forwarded as-is; run the smoke test before every real launch,
# and the profiling run before committing to a Slurm --time.

#SBATCH --job-name=deepfake_moe_avr_ssl
#SBATCH --gres=gpu:a40:4
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/avr_ssl_%j.out
#SBATCH --error=logs/avr_ssl_%j.err

set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "${repo_root}/pyproject.toml" ]] || repo_root="${SLURM_SUBMIT_DIR:?repo root not found next to this script and SLURM_SUBMIT_DIR is unset}"
cd "$repo_root"
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

args=()
extra_set=()
for arg in "$@"; do
  case "$arg" in
    --smoke-test)
      # Confirm the job runs end-to-end on the real checkpoint: one clip per
      # split, one optimizer step, tiny frame count, an eval + a checkpoint
      # save. Run this before every real launch.
      extra_set+=(
        --set train_samples_per_dataset=1 --set val_samples_per_dataset=1
        --set max_steps=1 --set gradient_accumulation_steps=1
        --set eval_steps=1 --set num_frames=4
      )
      ;;
    --profile)
      # 10 steps, real num_frames, eval disabled (eval_steps set past
      # max_steps) — use train_runtime / 10 to size the Slurm --time.
      extra_set+=(
        --set max_steps=10 --set num_frames=32
        --set eval_steps=1000 --set gradient_accumulation_steps=1
      )
      ;;
    *)
      args+=("$arg")
      ;;
  esac
done

uv run python -m deepfake_moe.run_train_avr_ssl "${args[@]}" "${extra_set[@]}"
