#!/bin/bash
# Train a registered detector. Same file runs locally or on Slurm:
#
#   bash   scripts/train.sh --config configs/models/unite/config.yaml --smoke-test
#   bash   scripts/train.sh --config configs/models/unite/config.yaml --profile
#   sbatch scripts/train.sh --config configs/models/unite/config.yaml
#   sbatch scripts/train.sh --config configs/models/unite/config.yaml \
#            --set loss.lambda_ad=0 loss.ad.enabled=false log.run_name=ff_ce
#
#SBATCH --job-name=mars_train
#SBATCH --gres=gpu:a40:4
#SBATCH --mem=164G
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "${repo_root}/pyproject.toml" ]] || repo_root="${SLURM_SUBMIT_DIR:?repo root not found next to this script and SLURM_SUBMIT_DIR is unset}"
cd "$repo_root"
mkdir -p logs checkpoints results

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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HOME="${HF_HOME:-${repo_root}/.cache/huggingface}"
# Footprint reports (ai-impact-accounting): grid region and its carbon intensity in
# kgCO2/kWh. PUE and water use fall back to the package defaults unless DIA_PUE and
# DIA_WUE are set.
export DIA_REGION="${DIA_REGION:-ca-on}"
export DIA_CI="${DIA_CI:-0.03}"

args=()
extra_set=()
for arg in "$@"; do
  case "$arg" in
    --smoke-test)
      extra_set+=(
        --set data.test=ffpp
        --set data.manifest=artifacts/ffpp/manifest_smoke.csv
        --set data.max_videos_per_class=8 --set data.n_frames=16
        --set train.epochs=1 --set train.max_steps=4
        --set train.micro_batch_size=2 --set data.num_workers=4
        --set log.wandb=false --set log.run_name=smoke_train
      )
      ;;
    --profile)
      extra_set+=(
        --set data.test=ffpp
        --set data.manifest=artifacts/ffpp/manifest_profile.csv
        --set data.max_videos_per_class=25
        --set train.epochs=1 --set train.max_steps=10
        --set log.wandb=false --set log.run_name=profile
      )
      ;;
    *)
      args+=("$arg")
      ;;
  esac
done

gpus="$(nvidia-smi -L 2>/dev/null | wc -l || true)"
[[ "${gpus:-0}" -ge 1 ]] || gpus=1
read -r -a runner <<< "${UNITE_RUNNER-uv run}"

echo "repo_root=${repo_root}"
echo "launching on ${gpus} GPU(s): ${args[*]} ${extra_set[*]}"
# Loopback rendezvous instead of --standalone: on some nodes the hostname torchrun
# looks up does not resolve. The port is per job so two jobs on one node do not clash.
"${runner[@]}" torchrun \
  --nnodes=1 --master_addr=127.0.0.1 --master_port="$((29500 + ${SLURM_JOB_ID:-0} % 1000))" \
  --nproc_per_node="${gpus}" \
  -m mars.run_train "${args[@]}" "${extra_set[@]}"
