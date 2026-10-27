#!/bin/bash
# Evaluate a registered detector. Same file runs locally or on Slurm:
#
#   bash   scripts/eval.sh --config configs/models/unite/config.yaml \
#            --checkpoint checkpoints/unite/ff_ce_ad/best.pt --split test
#   sbatch scripts/eval.sh --config configs/models/unite/config.yaml \
#            --checkpoint checkpoints/unite/ff_ce_ad/best.pt
#
#SBATCH --job-name=mars_eval
#SBATCH --gres=gpu:a40:4
#SBATCH --mem=164G
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${repo_root}/.cache/huggingface}"
# Footprint reports (ai-impact-accounting): grid region and its carbon intensity in
# kgCO2/kWh. PUE and water use fall back to the package defaults unless DIA_PUE and
# DIA_WUE are set.
export DIA_REGION="${DIA_REGION:-ca-on}"
export DIA_CI="${DIA_CI:-0.03}"

gpus="$(nvidia-smi -L 2>/dev/null | wc -l || true)"
[[ "${gpus:-0}" -ge 1 ]] || gpus=1
read -r -a runner <<< "${UNITE_RUNNER-uv run}"

# Per-GPU eval batch size (forward-only, so well above the training micro batch).
# Passed first so any --set the caller adds still wins.
eval_bs="${EVAL_BATCH_SIZE:-16}"

# run_split holds one DataLoader (num_workers each, persistent) per test corpus at
# once, times the DDP ranks. On the RAM-poor a40 nodes that pool OOM-kills the
# workers, so keep it small for eval regardless of the training setting.
eval_workers="${EVAL_NUM_WORKERS:-2}"

echo "repo_root=${repo_root}"
# Loopback rendezvous instead of --standalone: on some nodes the hostname torchrun
# looks up does not resolve. The port is per job so two jobs on one node do not clash.
"${runner[@]}" torchrun \
  --nnodes=1 --master_addr=127.0.0.1 --master_port="$((29500 + ${SLURM_JOB_ID:-0} % 1000))" \
  --nproc_per_node="${gpus}" \
  -m mars.run_eval \
  --set "train.eval_batch_size=${eval_bs}" "data.num_workers=${eval_workers}" "$@"
