#!/bin/bash
# Phase 1 eval. The same file runs locally or on Slurm:
#
#   bash   scripts/eval.sh --dry-run
#   sbatch scripts/eval.sh --experiment-name Qwen3-Baseline --model qwen3-omni
#   sbatch --array=0-4 scripts/eval.sh --experiment-name Qwen3-Baseline --model qwen3-omni
#
# Sharding is taken from the Slurm array size; results go to
# results/<experiment-name>/<model>/. Each shard writes its own
# shard_XXX/predictions.jsonl, and the metrics are aggregated automatically:
#   * single job              -> metrics written at the end of the run
#   * Slurm array (sharded)   -> the first shard queues an afterok metrics job
#
# To (re)compute metrics from existing shard predictions yourself:
#   bash scripts/eval.sh --score-only --experiment-name <name> --model <model>
#
# Predict/score are separate, resumable stages — any extra flags are forwarded
# to run_eval: --predict-only (inference only, no report), --resume-predictions
# (skip clips already in predictions.jsonl), --retry-failed (rerun only failed
# clips), --score-only (metrics from existing predictions.jsonl, no model load).
#
#SBATCH --job-name=deepfake_moe_eval
#SBATCH --gres=gpu:a40:4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH --array=0-0

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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1  # stream logs to the .out file in real time (no block buffering)

# Optional: point the manifest src_path prefix at the on-disk raw data.
if [[ -n "${MANIFEST_DATA_ROOT:-}" && -n "${RAW_DATA_DIR:-}" ]]; then
  for name in ${DATASET_SYMLINKS:-}; do
    if [[ -d "${RAW_DATA_DIR}/${name}" && ! -e "${MANIFEST_DATA_ROOT}/${name}" ]]; then
      ln -s "${RAW_DATA_DIR}/${name}" "${MANIFEST_DATA_ROOT}/${name}"
    fi
  done
fi

uv run python -m deepfake_moe.run_eval \
  --num-shards "${SLURM_ARRAY_TASK_COUNT:-1}" \
  --shard-id "${SLURM_ARRAY_TASK_ID:-0}" \
  "$@"

# Sharded array: the first shard queues a CPU-only metrics job that waits for the
# whole array (afterok), then aggregates all shards into one report.
if [[ "${SLURM_ARRAY_TASK_COUNT:-1}" -gt 1 && "${SLURM_ARRAY_TASK_ID:-0}" -eq 0 ]]; then
  sbatch --dependency="afterok:${SLURM_ARRAY_JOB_ID}" \
    --job-name=deepfake_moe_metrics --cpus-per-task=4 --mem=16G --time=00:30:00 \
    --output=logs/metrics_%j.out \
    --wrap="cd '${PWD}' && uv run python -m deepfake_moe.run_eval --score-only $*"
fi
