#!/bin/bash
# Faithfulness tests for trained MARS on one GPU:
#
#   sbatch scripts/faithfulness.sh --config configs/models/mars/config.yaml \
#     --checkpoint checkpoints/mars/main_ffpp_gtav/mars_s0/best.pt --test splice
#
#SBATCH --job-name=mars_faithfulness
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/faithfulness_%j.out
#SBATCH --error=logs/faithfulness_%j.err
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "${repo_root}/pyproject.toml" ]] || repo_root="${SLURM_SUBMIT_DIR:?repo root not found next to this script and SLURM_SUBMIT_DIR is unset}"
cd "$repo_root"
mkdir -p logs results
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
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${repo_root}/.cache/huggingface}"
read -r -a runner <<< "${UNITE_RUNNER-uv run}"
"${runner[@]}" python -m mars.run_faithfulness "$@"
