#!/bin/bash
# Main table: FF++ + GTA-V, 3 epochs, seed 0 (group main_ffpp_gtav).
#   bash experiments/main.sh [models...]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
models_from_args "$@"
for model in "${MODELS[@]}"; do
  submit "$model" 3 main_ffpp_gtav "${model}_s0"
done
