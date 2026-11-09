#!/bin/bash
# Shared submit helper for the experiment scripts in this folder.
#
# Each script loops over the models given as arguments (default: all 8) and
# submits one scripts/train.sh job per run. DRY_RUN=1 prints the sbatch lines
# instead of submitting them.
set -euo pipefail

ALL_MODELS=(mars unite gend forada dfd_hr demamba effort restrav)

# Training hours per epoch, from the 3-epoch main table.
declare -A HOURS_PER_EPOCH=(
  [mars]=1.1 [unite]=1.3 [gend]=1.9 [forada]=2.1
  [dfd_hr]=6.8 [demamba]=3.0 [effort]=1.0 [restrav]=1.0
)

# Maximum run time of the Slurm partitions, in hours.
TIERS=(3 12 24 72 168)

# Slurm time limit for a run: the expected time (1.6 x training plus 4 h for the test
# battery), rounded up to the limit of the partition it lands in, so a run gets that
# partition's full time at no cost in queue.
time_limit() {
  local model="$1" epochs="$2"
  local hours tier
  hours=$(awk -v h="${HOURS_PER_EPOCH[$model]}" -v e="$epochs" 'BEGIN { t = h * e * 1.6 + 4; print (t == int(t)) ? t : int(t) + 1 }')
  for tier in "${TIERS[@]}"; do
    if ((hours <= tier)); then
      hours=$tier
      break
    fi
  done
  ((hours > 168)) && hours=168
  if ((hours >= 24)); then
    printf '%d-%02d:00:00' $((hours / 24)) $((hours % 24))
  else
    printf '%02d:00:00' "$hours"
  fi
}

# submit <model> <epochs> <group> <run_name> [--set overrides...]; CONFIG overrides the YAML.
submit() {
  local model="$1" epochs="$2" group="$3" run_name="$4"
  shift 4
  local cmd=(sbatch --time="$(time_limit "$model" "$epochs")"
    scripts/train.sh --config "${CONFIG:-configs/models/${model}/config.yaml}"
    --experiment "$run_name" --set "log.group=${group}" "$@")
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "${cmd[*]}"
  else
    "${cmd[@]}"
  fi
}

# models_from_args "$@": the given models, or all 8.
models_from_args() {
  if (($#)); then
    MODELS=("$@")
  else
    MODELS=("${ALL_MODELS[@]}")
  fi
  for model in "${MODELS[@]}"; do
    [[ -n "${HOURS_PER_EPOCH[$model]:-}" ]] || { echo "unknown model: $model" >&2; exit 1; }
  done
}

cd "$(dirname "${BASH_SOURCE[0]}")/.."
