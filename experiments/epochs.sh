#!/bin/bash
# Epochs 5, 10, 20 with the lr decay stretched to match (decay_steps = 1000 x epochs / 3),
# so every budget ends at the same lr; 3 is the main-table run (group epochs).
#   bash experiments/epochs.sh [models...]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
models_from_args "$@"
declare -A DECAY=([5]=1667 [10]=3333 [20]=6667)
for model in "${MODELS[@]}"; do
  for epochs in 5 10 20; do
    submit "$model" "$epochs" epochs "${model}_ep${epochs}_s0" "train.epochs=${epochs}" "train.decay_steps=${DECAY[$epochs]}"
  done
done
