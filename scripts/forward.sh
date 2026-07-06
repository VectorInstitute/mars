#!/bin/bash
# One raw Qwen3-Omni forward pass (inspection). The same file runs locally or on Slurm:
#
#   bash   scripts/forward.sh --video clip.mp4 --audio clip.wav --generate
#   sbatch scripts/forward.sh --model qwen3-omni-hf --video /path/clip.mp4
#
# Extra flags are forwarded to run_forward (--model defaults to qwen3-omni-hf).
#
#SBATCH --job-name=deepfake_moe_forward
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/forward_%j.out

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
export PYTHONUNBUFFERED=1  # stream logs to the .out file in real time (no block buffering)

uv run python -m deepfake_moe.run_forward "$@"
