#!/bin/bash
# Extra seeds; seed 0 is the main-table run (group seeds).
#   bash experiments/seeds.sh [models...]               seeds 5 and 10
#   SEEDS="15 20" bash experiments/seeds.sh [models...]  other seeds
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
models_from_args "$@"
read -r -a seeds <<< "${SEEDS:-5 10}"
for model in "${MODELS[@]}"; do
  for seed in "${seeds[@]}"; do
    submit "$model" 3 seeds "${model}_s${seed}" "train.seed=${seed}"
  done
done
